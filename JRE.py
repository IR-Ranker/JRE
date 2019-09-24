#encoding:utf-8
from config import *
import os, math, shutil, time
import tensorflow as tf
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from datetime import datetime
from AlexNet import AlexNet
import cPickle
import sys
reload(sys)
sys.setdefaultencoding('utf8')

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

np.random.seed(2017)
tf.set_random_seed(2017)

learning_rate = 0.0001
dropout_rate = 1.


def image(image_placeholders):
    [x, vr_type, keep_prob] = image_placeholders

    voca_vr_type = tf.Variable(tf.ones([type_num, feature_map_height, feature_map_width]), dtype=tf.float32, name="image_vr_type")  
    attention_map = tf.einsum('ij,jkl->ikl', vr_type, voca_vr_type)
    attention_map = tf.expand_dims(attention_map, 3)
    with tf.name_scope('image'):
        model = AlexNet(x, keep_prob, num_class, train_layers, weights_path=model_base+'bvlc_alexnet.npy')
    
        feature = tf.multiply(model.pool5, attention_map)
        feature = model.pool5
        shape = int(np.prod(feature.get_shape()[1:]))
        feature_flattened = tf.reshape(feature, [-1, shape])
        fc6 = fc(feature_flattened, shape, 4096, name='fc6', relu='relu')
        dropout6 = dropout(fc6, dropout_rate, name='dropout_fc6')
        fc7 = fc(dropout6, 4096, 1000, name = 'fc7', relu='relu')
        dropout7 = dropout(fc7, dropout_rate, name='dropout_fc7')
        image_feature = fc(dropout7, 1000, 1, name = 'fc8', relu='no')
        pred_image = tf.nn.sigmoid(image_feature)
    return pred_image

def text(text_placeholders, text_type):
    if text_type == 'title':
        [text, text_len, attention_text, session, sess_len] = text_placeholders
    elif text_type == 'snippet':
        [text, text_len, attention_text, session, sess_len, sessions_weight] = text_placeholders

    voca = load_voca()
    voca_embed = tf.Variable(voca, trainable = True, dtype=tf.float32, name='voca_embed_'+text_type)
    text_embed = tf.nn.embedding_lookup(voca_embed, text)
    sess_embed = tf.nn.embedding_lookup(voca_embed, session)

    text_outputs, text_last_states = text_net(text_type, text_len, text_embed)
    if lstm_mode == 'mean':
        text_output = tf.multiply(text_outputs, attention_text)
        text_output = tf.reduce_mean(text_output, 1)
    elif lstm_mode == 'final':
        text_output =  [0]*batch_size
        for i in range(batch_size):
            text_output[i] = text_outputs[i, text_len[i]-1, :]
        text_output = tf.stack(text_output)
    text_feature_lstm = text_output

    sess_outputs, sess_last_states = text_net('sess_'+text_type, sess_len, sess_embed)
    if lstm_mode == 'mean':
        if text_type == 'title':
            sess_output = tf.reduce_mean(sess_outputs, 1)
        elif text_type == 'snippet':
            sess_output = tf.multiply(sess_outputs, sessions_weight)
            sess_output = tf.reduce_mean(sess_output, 1)
    elif lstm_mode == 'final':
        sess_output =  [0]*batch_size
        for i in range(batch_size):
            sess_output[i] = sess_outputs[i, sess_len[i]-1, :]
        sess_output = tf.stack(sess_output)
    sess_feature_lstm = sess_output
    print('text net seted')


    cls1_text = fc(text_feature_lstm, 1000, 1000, name = 'cls1_'+text_type)
    cls1_text_dropout = tf.nn.dropout(cls1_text, dropout_rate, name = 'cls1_dropout_'+text_type)
    text_feature = cls1_text_dropout

    cls1_sess = fc(sess_feature_lstm, 1000, 1000, name = 'cls1_sess_'+text_type)
    cls1_sess_dropout = tf.nn.dropout(cls1_sess, dropout_rate, name = 'cls1_dropout_sess_'+text_type)
    sess_feature = cls1_sess_dropout

    text_output_norm = tf.sqrt(tf.reduce_sum(tf.square(text_feature), 1))
    sess_output_norm = tf.sqrt(tf.reduce_sum(tf.square(sess_feature), 1)) 
    cosin = tf.divide(tf.reduce_sum(tf.multiply(text_feature, sess_feature), 1), tf.multiply(text_output_norm, sess_output_norm))
    pred_text = tf.expand_dims(tf.nn.sigmoid(cosin), 1) 
    return pred_text

def html(html_placeholders):
    [html_tag, html_class] = html_placeholders

    voca_embed_tag = tf.Variable(tf.random_normal([tag_num, embedding_size], stddev=0.1), dtype=tf.float32, name="html_tag_embed")
    voca_embed_class = tf.Variable(tf.random_normal([class_num, embedding_size], stddev=0.1), dtype=tf.float32, name="html_class_embed")
  
    tag_embed = tf.expand_dims(tf.nn.embedding_lookup(voca_embed_tag, html_tag), 3)
    class_embed = tf.expand_dims(tf.nn.embedding_lookup(voca_embed_class, html_class), 3)

    h_pool_flat_tag = html_cnn('tag', tag_embed)
    h_pool_flat_class = html_cnn('class', class_embed)
    html = tf.multiply(h_pool_flat_tag, h_pool_flat_class)
    
    html_fc1 = fc(html, num_filters * len(filter_sizes), 1000, name = 'html_fc1', relu='relu')
    html_fc2 = fc(html_fc1, 1000, 1000, name = 'html_fc2', relu='relu')
    html_fc3 = fc(html_fc2, 1000, 1, name = 'html_fc3', relu='no')
    html_feature = html_fc3
    
    pred_html = tf.nn.sigmoid(html_feature)
    return pred_html
    

def train(batch_size, num_epochs, num_train, num_val, alpha_regularizer, lstm_mode, html_type):
    display_step = 10
    filewriter_path = model_base+"tensorboard/"
    checkpoint_path = model_base+"checkpoint/"
    if os.path.exists(filewriter_path):  
        shutil.rmtree(filewriter_path)
    os.makedirs(filewriter_path)
    if not os.path.isdir(checkpoint_path): os.makedirs(checkpoint_path)

    #share placeholders
    keep_prob = tf.placeholder(tf.float32, name='keep_prob_placeholder')
    y = tf.placeholder(tf.float32, [None,], name='label_placeholder')

    #image placeholders
    x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='image_placeholder')
    vr_type = tf.placeholder(tf.float32, [None, type_num], name='type_placeholder')

    #text placeholders
    title = tf.placeholder(tf.int32, (None, None))
    title_len = tf.placeholder(tf.int32, (None)) 
    snippet = tf.placeholder(tf.int32, (None, None))
    snippet_len = tf.placeholder(tf.int32, (None)) 
    session_title = tf.placeholder(tf.int32, (None, None))
    sess_len_title = tf.placeholder(tf.int32, (None)) 
    session_snippet = tf.placeholder(tf.int32, (None, None))
    sess_len_snippet = tf.placeholder(tf.int32, (None)) 
    sessions_weight_snippet = tf.placeholder(tf.float32, [None, sess_sen_len_snippet, feature_dim])
    attention_title = tf.placeholder(tf.float32, [None, max_title_len_top, feature_dim])
    attention_snippet = tf.placeholder(tf.float32, [None, max_snippet_len_top, feature_dim])

    #html placeholders
    html_tag = tf.placeholder(tf.int32, [None, html_dim], name='tag_placeholder')
    html_class = tf.placeholder(tf.int32, [None, html_dim], name='class_placeholder')

    #with tf.name_scope('image'):
    image_placeholders = [x, vr_type, keep_prob]
    pred_image = image(image_placeholders)

    #with tf.name_scope('title'):
    title_placeholders = [title, title_len, attention_title, session_title, sess_len_title]
    pred_title = text(title_placeholders, 'title')

    #with tf.name_scope('snippet'):
    snippet_placeholders = [snippet, snippet_len, attention_snippet, session_snippet, sess_len_snippet, sessions_weight_snippet]
    pred_snippet = text(snippet_placeholders, 'snippet')

    #with tf.name_scope('html'):
    html_placeholders = [html_tag, html_class]
    pred_html = html(html_placeholders)

    #fusion
    pred_combine = tf.squeeze(tf.concat([pred_image, pred_title, pred_snippet, pred_html], 1))
    balance_raw = tf.Variable(tf.ones([4]), name='balance', trainable=True)
    #without XPN
    #pred_combine = tf.squeeze(tf.concat([pred_image, pred_title, pred_html], 1))
    #balance_raw = tf.Variable(tf.ones([3]), name='balance', trainable=True)

    balance_sum = tf.reduce_sum(balance_raw)
    balance = tf.div(balance_raw, balance_sum)
    
    pred_final = tf.reduce_sum(tf.multiply(pred_combine, balance), 1)

    with tf.name_scope("loss"):
      regularizer = tf.contrib.layers.l2_regularizer(alpha_regularizer)
      loss_regularizer = tf.contrib.layers.apply_regularization(regularizer, tf.trainable_variables())
      
      #sigmoid_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.squeeze(y), logits = pred_final)
      sigmoid_cross_entropy = cross_entropy(labels = tf.squeeze(y), logits = pred_final)
      loss_cross_entropy = tf.reduce_mean(sigmoid_cross_entropy, name='loss_cross_entropy')
      loss_mse = tf.reduce_mean(tf.square(pred_final - tf.squeeze(y)))
      
      loss = loss_cross_entropy

    print('Get ready! We are going to print all the trainable vars.')
    var_list = [v for v in tf.trainable_variables()]
    for var in var_list:
        print(var.name)
    print('Ok, print done.')

    var_train_list = var_list
    with tf.name_scope("train"):
      gradients = tf.gradients(loss, var_train_list)
      #gradients, global_norm = tf.clip_by_global_norm(gradients, 1) 
      gradients = list(zip(gradients, var_train_list)) 
      #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
      optimizer = tf.train.AdamOptimizer(learning_rate)
      train_op = optimizer.apply_gradients(grads_and_vars=gradients)
      #train_op = optimizer.minimize(loss)

    for var in var_list:
      tf.summary.histogram(var.name, var)    
    tf.summary.scalar('loss_regularizer_fusion', loss_regularizer)
    tf.summary.scalar('loss_cross_entropy_fusion', loss_cross_entropy)
    tf.summary.scalar('loss_mse_fusion', loss_mse)
    tf.summary.scalar('loss_fusion', loss)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(filewriter_path)


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    sess.run(tf.global_variables_initializer()) 
    writer.add_graph(sess.graph)    
    images_vals, title_vars, snippet_vars, html_vars = [], [], [], []
    for var in var_list:
        if var.name.find("title") != -1:
            print('title:  '+var.name)
            title_vars.append(var)
        elif var.name.find("snippet") != -1:
            print('snippet:  '+var.name)
            snippet_vars.append(var)
        elif var.name.find('html') != -1:
            print('html:  '+var.name)
            html_vars.append(var)
        elif var.name.find("balance") == -1:
            print('image:  '+var.name)
            images_vals.append(var)

    # saver_image = tf.train.Saver(images_vals)
    # saver_title = tf.train.Saver(title_vars) 
    # saver_snippet = tf.train.Saver(snippet_vars)
    # saver_html = tf.train.Saver(html_vars) 
    # saver_image.restore(sess, model_base+"checkpoint/VPN/model_image.ckpt")
    # print('image model successfully loaded!')
    # saver_title.restore(sess, model_base+"checkpoint/TSN/model_title.ckpt")
    # print('title model successfully loaded!')
    # saver_snippet.restore(sess, model_base+"checkpoint/SSN/model_snippet.ckpt")
    # print('snippet model successfully loaded!')
    # saver_html.restore(sess, model_base+'checkpoint/HSN/model_html.ckpt')
    # print('html model successfully loaded!')
    saver = tf.train.Saver(max_to_keep=20)

    train_dataset = val_dataset = '201709'
    train_path = data_base+'201709/info_top_10_id_201709'
    val_path = data_base+'201709/info_top_10_id_201709'
    images_train, rels_train, num_data_train = set_data_image(train_path, train_dataset)
    images_val, rels_val, num_data_val = set_data_image(val_path, val_dataset)
    type_train = set_data_type(train_path)
    type_val = set_data_type(val_path)
    titles_train, snippets_train, rels_train, queries_train, num_data_train = set_data_text('text', train_path)
    titles_val, snippets_val, rels_val, queries_val, num_data_val = set_data_text('text', val_path)
    sess_title_train, sessions_weight_title_train = set_data_sess('title', train_path, train_dataset)
    sess_snippet_train, sessions_weight_snippet_train = set_data_sess('snippet', train_path, train_dataset)
    sess_title_val, sessions_weight_title_val = set_data_sess('title', val_path, val_dataset)
    sess_snippet_val, sessions_weight_snippet_val = set_data_sess('snippet', val_path, val_dataset)
    DFS_tag_train, DFS_class_train, BFS_tag_train, BFS_class_train, rels_train, num_data_train = set_data_html(train_path, train_dataset)
    DFS_tag_val, DFS_class_val, BFS_tag_val, BFS_class_val, rels_val, num_data_val = set_data_html(val_path, val_dataset)
 
    print('train data num:{}'.format(num_data_train))
    print('val data num:{}'.format(num_data_val))


    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(), filewriter_path))   
    if num_train == 'all':
        num_train = num_data_train
    else:
        num_train = int(num_train)  
    if num_val == 'all':
        num_val = num_data_val
    else:
        num_val = int(num_val)

    for epoch in range(num_epochs):       
        print("{} Epoch number: {}".format(datetime.now(), epoch+1))
        step = 1
        for iter in xrange(num_train / batch_size):
            ind = set_random_ind(num_data_train, batch_size, random = True)
            pic_input, label_input = data_batch_image(images_train, rels_train, num_data_train, batch_size, ind)
            type_input = data_batch_type(type_train, batch_size, ind)
            title_input, title_len_input, label_input, attention_title_input = data_batch_text(titles_train, queries_train, window_weight, rels_train, num_data_train, batch_size, max_title_len_top, ind)
            snippet_input, snippet_len_input, label_input, attention_snippet_input = data_batch_text(snippets_train, queries_train, window_weight, rels_train, num_data_train, batch_size, max_snippet_len_top, ind)
            sess_title_input, sess_title_len_input, label_input, attention_sess_title_input = data_batch_text(sess_title_train, queries_train, window_weight, rels_train, num_data_train, batch_size, sess_sen_len_title, ind)
            sess_snippet_input, sess_snippet_len_input, label_input, attention_sess_snippet_input = data_batch_text(sess_snippet_train, queries_train, window_weight, rels_train, num_data_train, batch_size, sess_sen_len_snippet, ind)
            sessions_weight_snippet_input = sess_weight_batch('snippet', batch_size, sessions_weight_snippet_train, ind)
            if html_type=='DFS':
                tag_input, label_input = data_batch_html(DFS_tag_train, rels_train, ind)
                class_input, label_input = data_batch_html(DFS_class_train, rels_train, ind)
            elif html_type=='BFS':
                tag_input, label_input = data_batch_html(BFS_tag_train, rels_train, ind)
                class_input, label_input = data_batch_html(BFS_class_train, rels_train, ind)

            train_op_, loss_, loss_cross_entropy_, loss_mse_, loss_regularizer_, merged_summary_, pred_final_, pred_combine_, balance_ = sess.run([train_op, loss, loss_cross_entropy, loss_mse, loss_regularizer, merged_summary, pred_final, pred_combine, balance], 
                feed_dict={
                    y: label_input, keep_prob: dropout_rate,
                    x: pic_input, vr_type: type_input, 
                    title: title_input, title_len: title_len_input, session_title: sess_title_input, sess_len_title: sess_title_len_input, attention_title: attention_title_input,
                    snippet: snippet_input, snippet_len: snippet_len_input, session_snippet: sess_snippet_input, sess_len_snippet: sess_snippet_len_input, sessions_weight_snippet: sessions_weight_snippet_input, attention_snippet: attention_snippet_input,
                    html_tag: tag_input, html_class: class_input
                    })
            print("the "+str(epoch+1)+'th epoch, '+str(iter+1)+'th batch:  loss:{}  loss_cross_entropy:{}  loss_mse:{}  loss_regularizer:{}'.format(loss_, loss_cross_entropy_, loss_mse_, loss_regularizer_))            
            print(balance_)

            if step%display_step == 0:
                writer.add_summary(merged_summary_, epoch*num_train/batch_size + step)               
            step += 1

        dropout_rate_val = 1
        print("{} Start validation...".format(datetime.now()))
        loss_total = 0.
        pred_all, label_all = [], []
        iters = num_val/batch_size
        for iter in xrange(iters):
            ind = set_random_ind(num_data_val, batch_size, random = False, iter_ = iter)
            pic_input, label_input = data_batch_image(images_val, rels_val, num_data_val, batch_size, ind)
            type_input = data_batch_type(type_val, batch_size, ind)
            title_input, title_len_input, label_input, attention_title_input = data_batch_text(titles_val, queries_val, window_weight, rels_val, num_data_val, batch_size, max_title_len_top, ind)
            snippet_input, snippet_len_input, label_input, attention_snippet_input = data_batch_text(snippets_val, queries_val, window_weight, rels_val, num_data_val, batch_size, max_snippet_len_top, ind)
            sess_title_input, sess_title_len_input, label_input, attention_sess_title_input = data_batch_text(sess_title_val, queries_val, window_weight, rels_val, num_data_val, batch_size, sess_sen_len_title, ind)
            sess_snippet_input, sess_snippet_len_input, label_input, attention_sess_snippet_input = data_batch_text(sess_snippet_val, queries_val, window_weight, rels_val, num_data_val, batch_size, sess_sen_len_snippet, ind)
            sessions_weight_snippet_input = sess_weight_batch('snippet', batch_size, sessions_weight_snippet_val, ind)


            if html_type=='DFS':
                tag_input, label_input = data_batch_html(DFS_tag_val, rels_val, ind)
                class_input, label_input = data_batch_html(DFS_class_val, rels_val, ind)
            elif html_type=='BFS':
                tag_input, label_input = data_batch_html(BFS_tag_val, rels_val, ind)
                class_input, label_input = data_batch_html(BFS_class_val, rels_val, ind)
            loss_, loss_cross_entropy_, loss_mse_, loss_regularizer_ = sess.run([loss, loss_cross_entropy, loss_mse, loss_regularizer], 
                feed_dict={
                    y: label_input, keep_prob: dropout_rate,
                    x: pic_input, vr_type: type_input, 
                    title: title_input, title_len: title_len_input, session_title: sess_title_input, sess_len_title: sess_title_len_input, attention_title: attention_title_input,
                    snippet: snippet_input, snippet_len: snippet_len_input, session_snippet: sess_snippet_input, sess_len_snippet: sess_snippet_len_input, sessions_weight_snippet: sessions_weight_snippet_input, attention_snippet: attention_snippet_input,
                    html_tag: tag_input, html_class: class_input
                    })
            loss_total += loss_*batch_size
            print("the "+str(epoch+1)+'th epoch, '+str(iter+1)+'th batch:  loss:{}  loss_cross_entropy:{}  loss_mse:{}  loss_regularizer:{}'.format(loss_, loss_cross_entropy_, loss_mse_, loss_regularizer_))        
            print('average loss: {}'.format(loss_total*1.0/iters/batch_size))

        print("{} Saving checkpoint of model...".format(datetime.now()))       
        checkpoint_name = os.path.join(checkpoint_path, 'model_JRE_epoch_'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)       
        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))



def test(batch_size, num_test, epoch_id, lstm_mod, html_type):
    #share placeholders
    keep_prob = tf.placeholder(tf.float32, name='keep_prob_placeholder')
    y = tf.placeholder(tf.float32, [None,], name='label_placeholder')

    #image placeholders
    x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='image_placeholder')
    vr_type = tf.placeholder(tf.float32, [None, type_num], name='type_placeholder')

    #text placeholders
    title = tf.placeholder(tf.int32, (None, None))
    title_len = tf.placeholder(tf.int32, (None)) 
    snippet = tf.placeholder(tf.int32, (None, None))
    snippet_len = tf.placeholder(tf.int32, (None)) 
    session_title = tf.placeholder(tf.int32, (None, None))
    sess_len_title = tf.placeholder(tf.int32, (None)) 
    session_snippet = tf.placeholder(tf.int32, (None, None))
    sess_len_snippet = tf.placeholder(tf.int32, (None)) 
    sessions_weight_snippet = tf.placeholder(tf.float32, [None, sess_sen_len_snippet, feature_dim])
    attention_title = tf.placeholder(tf.float32, [None, max_title_len_top, feature_dim])
    attention_snippet = tf.placeholder(tf.float32, [None, max_snippet_len_top, feature_dim])

    #html placeholders
    html_tag = tf.placeholder(tf.int32, [None, html_dim], name='tag_placeholder')
    html_class = tf.placeholder(tf.int32, [None, html_dim], name='class_placeholder')

    #with tf.name_scope('image'):
    image_placeholders = [x, vr_type, keep_prob]
    pred_image = image(image_placeholders)

    #with tf.name_scope('title'):
    title_placeholders = [title, title_len, attention_title, session_title, sess_len_title]
    pred_title = text(title_placeholders, 'title')

    #with tf.name_scope('snippet'):
    snippet_placeholders = [snippet, snippet_len, attention_snippet, session_snippet, sess_len_snippet, sessions_weight_snippet]
    pred_snippet = text(snippet_placeholders, 'snippet')

    #with tf.name_scope('html'):
    html_placeholders = [html_tag, html_class]
    pred_html = html(html_placeholders)

    pred_combine = tf.squeeze(tf.concat([pred_image, pred_title, pred_snippet, pred_html], 1))
    balance_raw = tf.Variable(tf.ones([4]), name='balance', trainable=True)
    balance_sum = tf.reduce_sum(balance_raw)
    balance = tf.div(balance_raw, balance_sum)
    pred_final = tf.reduce_sum(tf.multiply(pred_combine, balance), 1)

    with tf.name_scope("loss"):
      sigmoid_cross_entropy = cross_entropy(labels = tf.squeeze(y), logits = pred_final)
      loss_cross_entropy = tf.reduce_mean(sigmoid_cross_entropy, name='loss_cross_entropy')
      loss_mse = tf.reduce_mean(tf.square(pred_final - tf.squeeze(y)))
      
      loss = loss_mse

    sess = tf.InteractiveSession()  
    sess.run(tf.global_variables_initializer())  
    saver = tf.train.Saver() 
    saver.restore(sess, model_base+'checkpoint/JRE/model_JRE_epoch_'+epoch_id+'.ckpt')

    test_dataset = '201709'
    tvt_file = data_base+'201709/info_top_10_id_201709'
    images_test, rels_test, num_data_test = set_data_image(tvt_file, test_dataset) 
    type_test = set_data_type(tvt_file)
    titles_test, snippets_test, rels_test, queries_test, num_data_test = set_data_text('text', tvt_file)
    sess_title_test, sessions_weight_title_test = set_data_sess('title', tvt_file, test_dataset)
    sess_snippet_test, sessions_weight_snippet_test = set_data_sess('snippet', tvt_file, test_dataset)
    DFS_tag_test, DFS_class_test, BFS_tag_test, BFS_class_test, rels_test, num_data_test = set_data_html(tvt_file, test_dataset)
    print('test data num:{}'.format(num_data_test))


    if num_test == 'all':
        num_test = num_data_test
    else:
        num_test = int(num_test)  

    dropout_rate_test = 1
    print("{} Start testing...".format(datetime.now()))
    loss_total = 0.
    pred_all, pred_combine_all, label_all = [], [], []
    iters = num_test/batch_size
    print('Start......')
    start = time.time()
    for iter in xrange(iters):
        ind = set_random_ind(num_data_test, batch_size, random = False, iter_ = iter)
        pic_input, label_input = data_batch_image(images_test, rels_test, num_data_test, batch_size, ind)
        type_input = data_batch_type(type_test, batch_size, ind)
        title_input, title_len_input, label_input, attention_title_input = data_batch_text(titles_test, queries_test, window_weight, rels_test, num_data_test, batch_size, max_title_len_top, ind)
        snippet_input, snippet_len_input, label_input, attention_snippet_input = data_batch_text(snippets_test, queries_test, window_weight, rels_test, num_data_test, batch_size, max_snippet_len_top, ind)   
        sess_title_input, sess_title_len_input, label_input, attention_sess_title_input = data_batch_text(sess_title_test, queries_test, window_weight, rels_test, num_data_test, batch_size, sess_sen_len_title, ind)
        sess_snippet_input, sess_snippet_len_input, label_input, attention_sess_snippet_input = data_batch_text(sess_snippet_test, queries_test, window_weight, rels_test, num_data_test, batch_size, sess_sen_len_snippet, ind)
        sessions_weight_snippet_input = sess_weight_batch('snippet', batch_size, sessions_weight_snippet_test, ind)

        if html_type=='DFS':
            tag_input, label_input = data_batch_html(DFS_tag_test, rels_test, ind)
            class_input, label_input = data_batch_html(DFS_class_test, rels_test, ind)
        elif html_type=='BFS':
            tag_input, label_input = data_batch_html(BFS_tag_test, rels_test, ind)
            class_input, label_input = data_batch_html(BFS_class_test, rels_test, ind)
        pred_final_, pred_combine_, loss_, loss_cross_entropy_, loss_mse_, balance_ = sess.run([pred_final, pred_combine, loss, loss_cross_entropy, loss_mse, balance], 
            feed_dict={
                y: label_input, keep_prob: dropout_rate,
                x: pic_input, vr_type: type_input, 
                title: title_input, title_len: title_len_input, session_title: sess_title_input, sess_len_title: sess_title_len_input, attention_title: attention_title_input,
                snippet: snippet_input, snippet_len: snippet_len_input, session_snippet: sess_snippet_input, sess_len_snippet: sess_snippet_len_input, sessions_weight_snippet: sessions_weight_snippet_input, attention_snippet: attention_snippet_input,
                html_tag: tag_input, html_class: class_input
                })
        
        loss_total += loss_*batch_size
        pred_all.append(pred_final_)
        pred_combine_all.append(pred_combine_)
        label_all.append(label_input)
        
    end = time.time()
    print('Total Time:{}'.format(end-start))

    print('average loss: {}'.format(loss_total*1.0/iters/batch_size))
    pred_all = np.squeeze(np.concatenate((np.array(pred_all)), axis=0))
    label_all = np.squeeze(np.concatenate((np.array(label_all)), axis=0))

    fusion_file = open(result_base+'JRE_'+test_dataset+'_'+epoch_id+'.txt', 'w')
    for i in range(iters*batch_size):
        fusion_file.write(images_test[i].split('/')[-1]+'\t'+str(label_all[i])+'\t'+str(pred_all[i])+'\n')
    


if __name__ == "__main__":
    op = sys.argv[1]
    if op=='train':
        batch_size = int(sys.argv[2])
        num_epochs = int(sys.argv[3])
        num_train = sys.argv[4]   #all
        num_val = sys.argv[5]
        alpha_regularizer = float(sys.argv[6])
        lstm_mode = sys.argv[7]
        html_type = sys.argv[8]
        train(batch_size, num_epochs, num_train, num_val, alpha_regularizer, lstm_mode, html_type)
    elif op=='test':
        batch_size = int(sys.argv[2])
        num_test = sys.argv[3]   #all
        lstm_mode = sys.argv[4]
        html_type = sys.argv[5]
        epoch_id = sys.argv[6]
        test(batch_size, num_test, epoch_id, lstm_mode, html_type)

