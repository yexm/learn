#coding:utf-8

import collections
import tensorflow.contrib as contrib
import tensorflow as tf
import random
import numpy as np
import sys


##########
#预料读取
##########
content = ""
with open('belling_the_cat.txt') as f:
    content = f.read()
    f.close()
words = content.split()
# print(words)
# print(len(words))

############
#构建正反向字典
############

def 构建正反向字典(words):
    '''预料的ids化和构建正反向字典'''
    count = collections.Counter(words).most_common()
    #[(',', 14), ('the', 11), ('.', 8), ('and', 7), ('to', 6), ('said', 6)
    
    #正向字典
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    
    #反向字典
    reverse_dictionary = dict(zip(dictionary.values(), \
                                  dictionary.keys()))
    return  dictionary, reverse_dictionary
    
###################
#调用函数生成正、反向字典
###################    
dictionary, reverse_dictionary = 构建正反向字典(words)

##########
#建模
##########


#网络参数的构建
vocab_size = len(dictionary)
crop_size = len(words)
# vocab_size = len(words)
n_input = 3
n_hidden = 512
batch_size = 20

weight = tf.get_variable("weight_out", [2*n_hidden, vocab_size])
bias = tf.get_variable("bias_out", [vocab_size])

#LSTM模型的建立
def LSTM(x, weight, bias):
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(x, n_input, 1)
    rnn_cell_forward = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple = True, forget_bias = 1.0)
    # rnn_cell_backward = contrib.rnn.BasicLSTMCell(n_hidden, state_is_tuple = True, forget_bias = 1.0)
    rnn_cell_backward = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple = True, forget_bias = 1.0)
    # rnn_cell_backward = contrib.rnn.BasicLSTMCell(n_hidden, state_is_tuple = True, forget_bias = 1.0)
#     outputs_states_fw
#     outputs_states_bw
    output, _,  _= tf.nn.static_bidirectional_rnn(rnn_cell_forward, \
                                   rnn_cell_backward, x, \
                                   dtype = tf.float32)
    return tf.matmul(output[-1], weight) + bias
    
#############
#训练数据随机采样
#############
def 随机采样语料(offset):
    
    while offset + n_input > crop_size:
        random.randint(0, crop_size - n_input)
    #x值得确定    
    symbols_in_key = [[dictionary[str(words[i])]] \
                      for i in range(offset, offset+n_input)]
        
    #y值得确定
    symblos_out_onehot = np.zeros([vocab_size], dtype = float)
#     print(vocab_size)
#     print(dictionary[str(words[offset+n_input])])
#     sys.exit("90")
    symblos_out_onehot[dictionary[str(words[offset+n_input])]] = 1.0
    
    return symbols_in_key, symblos_out_onehot
    
######################################
#构建符合tensorflow计算的形参类型和损失函数
######################################
x = tf.placeholder(tf.float32, [None, n_input, 1])
y = tf.placeholder(tf.float32, [None, vocab_size])

pred = LSTM(x, weight, bias)
cost = tf.reduce_mean((tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = pred)))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(cost)
#############
#精确率计算
#############
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuary = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


############
#训练模型
############

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50000):
        
        x_train, y_train = [], []
        for j in range(batch_size):#随机采样 
            new_x, new_y = 随机采样语料(random.randint(0, crop_size))
            x_train.append(new_x)
            y_train.append(new_y)
        sess.run(optimizer, feed_dict = {x:np.array(x_train), \
                                         y:np.array(y_train)})
    
        
        if (i+1)%2 == 0:
            acc, out_pred = sess.run([accuary, pred], \
                                     feed_dict = {x:np.array(x_train), \
                                                  y:np.array(y_train)})
        
            ########################
            #用正反向字典映射输入和输出的值
            ########################  
            symbols_in = [reverse_dictionary[word_index] for word_index in x_train[0]]
            
            symbols_out = reverse_dictionary[int(np.argmax(y_train, 1)[0])]
            
            pred_out =  reverse_dictionary[int(np.argmax(out_pred, 1))]

            print('精确率是：%f'%acc)
            print('%s-[%s] vs [%s]'%(symbols_in, symbols_out, pred_out))
            










