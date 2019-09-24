#encoding:utf-8
import os, math, copy
import tensorflow as tf
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import sys
reload(sys)
sys.setdefaultencoding('utf8')

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

np.random.seed(2017)
tf.set_random_seed(2017)

#share params
position_num = 10
feature_dim = 1000   #the length of text feature and image feature

#image params
num_class = 1
IMAGE_WIDTH = 550
IMAGE_HEIGHT = 130
type_num = 19
feature_map_width = 16
feature_map_height = 2
kernal = 256
train_layers = []

#text params
text_dim = 200    #text embed size
max_title_len_top = 20    #the max number of words in a sentence（query or title）
max_snippet_len_top = 100
sess_sen_len_snippet = 10
sess_sen_len_title = 20
window_size = 3
window_weight = [1.8, 2, 1.8]

#html params
html_dim = 10
embedding_size = 200
filter_sizes = [1,2,3,4,5]
num_filters = 256
list_tag = ['div', 'h3', 'table', 'span', 'ul', 'p']
list_class = ['wrap', 'Title', 'Table', 'Box', 'info', 'list', 'img', 'txt', 'link', 'query', 'div', 'hint', 'None', 'fb', 'weibo', 'time']
tag_num = len(list_tag)
class_num = len(list_class)

#your paths here
srr_base = ''
data_base = ''
model_base = ''
result_base = ''
suffix = ''


#share funcs
def cross_entropy(labels, logits):
    ans = labels * ( -tf.log(logits) ) + (1 - labels) * ( -tf.log(1 - logits) )
    return ans

def set_random_ind(num_data, batch_size, random = True, iter_ = 0):
    if random:
        index = np.random.permutation(num_data)
        ind = index[0: batch_size]
    else:
        ind = np.array(range(iter_*batch_size, batch_size*(iter_+1)))
    return ind

def dropout(x, keep_prob, name):
  return tf.nn.dropout(x, keep_prob, name = name)

def leaky_relu(x, alpha=0., max_value=None):
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),
                             tf.cast(max_value, dtype=tf.float32))
    x -= tf.constant(alpha, dtype=tf.float32) * negative_part
    return x

def fc(x, num_in, num_out, name, relu = 'relu'):
  with tf.variable_scope(name) as scope:
    # Create tf variables for the weights and biases
    weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
    #weights = tf.get_variable('weights', initializer=tf.truncated_normal([num_in, num_out], dtype=tf.float32, stddev=1e-1), trainable=True)    
    biases = tf.get_variable('biases', [num_out], trainable=True)  
    # Matrix multiply weights and inputs and add bias
    act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)  
    if relu == 'relu':
      # Apply ReLu non linearity
      relu = tf.nn.relu(act)      
      return relu
    elif relu == 'tanh':
      return tf.tanh(act)
    elif relu == 'leaky':
      return leaky_relu(act, alpha=0.4)
    elif relu == 'no':
      return act
    
def set_voca():
    voca = {}
    voca_file = open(data_base+'voca.txt')
    for line in voca_file:
        line = line.strip('\n').split('\t')
        voca[line[0]] = line[1]
    return voca

def load_voca(path = data_base+'vocabulary.npy'):
    voca_embed = np.load(path)
    with tf.Session() as sess:
        voca_embed = tf.convert_to_tensor(voca_embed, dtype=tf.float32).eval()
    return voca_embed


#image funcs
def set_data_image(data_path, dataset):
    query_file = open(data_base+'query_id')
    id_query = {}
    for line in query_file:
        line = line.strip('\n').split('\t')
        id_query[line[1]] = line[0]
    print('query-id loaded')

    img_dir = srr_base+'crop_'+dataset+'/'
    in_file = open(data_path)
    images = []
    rels = []
    for line in in_file:
        line = line.strip().split('\t')
        query = id_query[line[0]]
        imgs_temp = [img_dir+query+'_'+ind+'.png' for ind in line[3].split(' ')]
        rels_temp = [float(item) for item in line[2].split(' ')]
        images.extend(imgs_temp)
        rels.extend(rels_temp)
    images = np.array(images)
    rels = np.array(rels)
    num_data = len(images)
    return images, rels, num_data

def set_data_type(data_path):  
    in_file = open(data_path)
    types = []
    for line in in_file:
        line = line.strip('\n').split('\t')
        types_temp = [int(type_id) for type_id in line[6].split(' ')]
        types.extend(types_temp)
    types = np.array(types)
    return types

def data_batch_image(images, rels, num_data, batch_size, ind):
    img_names_batch = images[ind]
    pic_input = []
    for name in img_names_batch:
        img = Image.open(name)
        img = img.convert('RGB')
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        img = np.array(img,dtype='float32')    
        pic_input.append(img)            
    pic_input = np.array(pic_input,dtype='float32')

    label_input = rels[ind].astype('float32')  
    return pic_input, label_input

def data_batch_type(types, batch_size, ind):
    type_temp = types[ind]
    type_input = np.zeros((batch_size, type_num))
    type_input[:, type_temp] = 1
    return type_input

def image_names(data_path):
    query_file = open(data_base+'query_id')
    id_query = {}
    for line in query_file:
        line = line.strip('\n').split('\t')
        id_query[line[1]] = line[0]
    print('query-id loaded')

    in_file = open(data_path)
    images = []
    for line in in_file:
        line = line.strip().split('\t')
        query = id_query[line[0]]
        imgs_temp = [img_dir+query+'_'+ind+'.png' for ind in line[3].split(' ')]
        images.extend(imgs_temp)
    images = np.array(images)
    return images


#text funcs
def set_data_sess(text_type, data_path, dataset):
    if text_type == 'snippet':
        session_file = open(data_base+dataset+'/query_session_'+str(sess_sen_len_snippet)+'_'+dataset)
    elif text_type == 'title':
        session_file = open(data_base+dataset+'/query_session_query_'+dataset)  
    id_session, id_session_weight = {}, {}

    for line in session_file:
        line = line.strip('\n').split('\t')
        id_session[line[0]] = [item for item in line[1].split(' ')]
        id_session_weight[line[0]] = [float(item) for item in line[2].split(' ')]
    in_file = open(data_path)
    sessions, sessions_weight = [], []

    for line in in_file:
        line = line.strip('\n').split('\t')
        sessions_temp = [id_session[line[0]] for i in range(position_num)]   
        sessions_weight_temp = [id_session_weight[line[0]] for i in range(position_num)]
        sessions.extend(sessions_temp)
        sessions_weight.extend(sessions_weight_temp)
    sessions_weight = np.array(sessions_weight)
    return sessions, sessions_weight

def set_data_text(text_type, data_path):
    query_file = open(data_base+'id_query2id_cut')
    title_file = open(data_base+'id_title2id_cut')
    snippet_file = open(data_base+'id_snippet2id_cut')

    id_query, id_session, id_session_weight, id_title, id_snippet = {}, {}, {}, {}, {}
    for line in query_file:
        line = line.strip('\n').split('\t')
        id_query[line[0]] = [int(item) for item in line[1].split(' ')]
    for line in title_file:
        line = line.strip('\n').split('\t')
        id_title[line[0]] = [item for item in line[1].split(' ')]
    for line in snippet_file:
        line = line.strip('\n').split('\t')
        id_snippet[line[0]] = [item for item in line[1].split(' ')]
    
    in_file = open(data_path)
    sessions, sessions_weight, titles, snippets, rels, queries = [], [], [], [], [], []
    for line in in_file:
        line = line.strip('\n').split('\t')
        titles_temp = [id_title[title_id] for title_id in line[4].split(' ')]
        snippets_temp = [id_snippet[snippet_id] for snippet_id in line[5].split(' ')]
        rels_temp = [float(item) for item in line[2].split(' ')]
        queries_temp = [id_query[line[0]] for i in range(position_num)]  
        titles.extend(titles_temp)
        snippets.extend(snippets_temp)
        rels.extend(rels_temp)
        queries.extend(queries_temp)
    rels = np.array(rels)
    num_data = len(rels)
    return titles, snippets, rels, queries, num_data

def attention_window(sen, query, window_weight):
    sen_len = len(sen)
    attention = np.ones(sen_len+window_size-1)
    bias = window_size/2
    for i in range(sen_len):
        if sen[i] in query:
            for j in range(window_size):
                attention[i+j] *= window_weight[j]
    attention_temp = attention[bias:-bias] 
    eye = np.ones(feature_dim) 
    attention = attention_temp.reshape([sen_len, 1]) * eye
    return attention

def sess_weight_batch(text_type, batch_size, sessions_weight, ind):
    eye = np.ones(feature_dim) 
    sessions_weight_temp = sessions_weight[ind]
    if text_type == 'title':
        weight = sessions_weight_temp.reshape([batch_size, sess_sen_len_title, 1]) * eye
    elif text_type == 'snippet':
        weight = sessions_weight_temp.reshape([batch_size, sess_sen_len_snippet, 1]) * eye
    return weight

def data_batch_text(data, queries, window_weight, rels, num_data, batch_size, max_data_len_top, ind):
    data_input = []
    data_len_input = []
    attention_input = []
    for i in ind:
        item = data[i]
        query = queries[i]
        sentence = copy.copy(item)
        if len(sentence)<max_data_len_top:
            data_len_input.append(len(sentence))
            while len(sentence)<max_data_len_top:
                sentence.append(0)
        else:
            data_len_input.append(max_data_len_top)
            sentence = sentence[:max_data_len_top]
        sentence = np.array(sentence).astype(np.int32)
        data_input.append(sentence)
        attention = attention_window(sentence, query, window_weight)
        attention_input.append(attention)
    data_input = np.array(data_input)
    data_len_input = np.array(data_len_input)
    attention_input = np.array(attention_input)
    label_input = rels[ind].astype('float32')   
    return data_input, data_len_input, label_input, attention_input

def text_net(text_name, text_len, text_embed):
    with tf.variable_scope('LSTM_'+text_name):
        text_cell = tf.contrib.rnn.BasicLSTMCell(num_units=feature_dim, state_is_tuple=True)
        text_outputs, text_last_states = tf.nn.dynamic_rnn(
            cell=text_cell,
            dtype=tf.float32,
            sequence_length=text_len,
            inputs=text_embed)
    return text_outputs, text_last_states


#html funcs
def set_data_html(data_path, dataset):
    in_file = open(data_path)
    html_file = open(data_base+dataset+'/query_html_id'+'_'+dataset)
    query_html = {}
    for line in html_file:
        line = line.strip('\n').split('\t')
        query_html[line[0]] = [line[1], line[2], line[3], line[4]]

    DFS_tag, DFS_class, BFS_tag, BFS_class = [], [], [], []
    rels = []
    for line in in_file:
        line = line.strip('\n').split('\t')
        query = line[0]
        html_data = query_html[query]
        DFS_tag_temp = [item.split(' ') for item in html_data[0].split('#')]
        DFS_class_temp = [item.split(' ') for item in html_data[1].split('#')]
        BFS_tag_temp = [item.split(' ') for item in html_data[2].split('#')]
        BFS_class_temp = [item.split(' ') for item in html_data[3].split('#')]
        rels_temp = [float(item) for item in line[2].split(' ')]
        DFS_tag.extend(DFS_tag_temp); DFS_class.extend(DFS_class_temp)
        BFS_tag.extend(BFS_tag_temp); BFS_class.extend(BFS_class_temp)
        rels.extend(rels_temp)
    DFS_tag = np.array(DFS_tag).astype('float32'); DFS_class = np.array(DFS_class).astype('float32')
    BFS_tag = np.array(BFS_tag).astype('float32'); BFS_class = np.array(BFS_class).astype('float32')
    rels = np.array(rels).astype('float32')
    num_data = len(rels)
    return DFS_tag, DFS_class, BFS_tag, BFS_class, rels, num_data

def data_batch_html(data, rels, ind):
    data_input = data[ind]
    label_input = rels[ind]
    return data_input, label_input

def html_cnn(tag_or_class, embed):
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.variable_scope('html_'+tag_or_class+'_'+str(filter_size)):
            # Convolution Layer
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_"+tag_or_class, trainable=True)
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b_"+tag_or_class, trainable=True)
            conv = tf.nn.conv2d(
                embed,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu"+tag_or_class)
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, html_dim - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            pooled_outputs.append(pooled)
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    return h_pool_flat
