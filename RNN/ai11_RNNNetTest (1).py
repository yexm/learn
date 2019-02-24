#coding:utf-8

import collections#统计词频
import tensorflow as tf
import random
import numpy as np
import sys


tf.random_seed(1221)
"""读取数据"""
content =  ""
with open('./belling_the_cat.txt') as f:
    content = f.read() 

words = content.split()

def 构建正反向字典(words):
    '''构建正反向字典'''
    count = collections.Counter(words).most_common()
#     print(count)
    """构建正向字典"""
    dictionary = dict()
    for word, _ in count:
        dictionary[word]= len(dictionary)
#     print(dictionary)
    """构建反向字典"""
    reverse_dictionary = dict(zip(dictionary.values(), \
                                  dictionary.keys()))
    """函数值是返回正向字典和反向字典"""
    return dictionary, reverse_dictionary
dictionary, reverse_dictionary=构建正反向字典(words)
"""开始构建网络(模型)参数"""
vocab_size = len(dictionary)
crop_size = len(words)######################
n_input = 3
n_hidden = 512
batch_size = 20
#构建多层感知机部分的参数
weight= tf.get_variable('weight_out', [2*n_hidden, vocab_size],
                        initializer= tf.random_normal_initializer)
bias = tf.get_variable('bias_out', [vocab_size], initializer = \
                       tf.random_normal_initializer)

def RNN(x, weight, bias):
    x = tf.reshape(x, [-1, n_input])# [1, 2, 3]
    x = tf.split(x, n_input, 1)# [[1], 
                               #  [2], 
                               #  [3]]
    rnn_cell_format = tf.nn.rnn_cell.BasicLSTMCell(n_hidden,\
                                                   state_is_tuple= True,
                                                   forget_bias = 1.0)
    rnn_cell_backmat = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, \
                                                    state_is_tuple=True,
                                                    forget_bias = 1.0)
    outputs, outputs_states_fw, outputs_states_bw = \
    tf.nn.static_bidirectional_rnn(rnn_cell_format, \
                                   rnn_cell_backmat, x, dtype = tf.float32)
#     print(outputs)
#     print(outputs_states_fw)
#     print(outputs_states_bw)
#     sys.exit("54")
    return tf.matmul(outputs[-1], weight)+bias

"""数据转换"""
def 随机采集语料(offset):
    '''样本打标'''
    '''确定样本词的起始位置'''
    while offset + n_input + 1 > crop_size:
        offset = random.randint(0, crop_size-n_input-1)
    """随机产生样本范围中的3个连续的词，并将其映射为数值"""
    symbols_in_key = [[dictionary[str(words[i])]] \
            for i in range(offset, offset+n_input)]
    '''lable值独热处理'''
    symbols_out_onehot = np.zeros([vocab_size], dtype = float)
    symbols_out_onehot[dictionary[str(words[offset+n_input])]] = 1.0
    return symbols_in_key, symbols_out_onehot
"""搭建损失函数"""
x = tf.placeholder(tf.float32, [None, n_input, 1])
y = tf.placeholder(tf.float32, [None, vocab_size])
pred = RNN(x, weight, bias)
cost = tf.reduce_mean(tf.nn.\
                      softmax_cross_entropy_with_logits(logits=\
                                                      pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate = 1e-5).minimize(cost)
"""构造准确率计算函数"""
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))
"""把准确率变为百分比"""
accuary = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                         
"""训练模型"""
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    save = tf.train.Saver()
    for i in range(50000):#50000
        x_train, y_train =[],[]
        for b in range(batch_size):
            new_x, new_y = 随机采集语料(random.randint(0, crop_size))
            """3+1"""
            x_train.append(new_x)
            y_train.append(new_y)
        _opt = sess.run(optimizer, \
                        feed_dict={x:np.array(x_train),\
                                   y:np.array(y_train)})
        if i%100==0:
            acc, out_pred = sess.run([accuary, pred], \
                                     feed_dict = \
                                     {x:np.array(x_train),\
                                      y:np.array(y_train)})
            symbols_in = [reverse_dictionary[word_index[0]] \
                          for word_index in x_train[0]]
            symbols_out = reverse_dictionary[int(np.argmax(y_train, 1)[0])]
            pred_out = reverse_dictionary[int(np.argmax(out_pred, 1)[0])]
            print('Acc:%f'%acc)
            print('%s-[%s]vs[%s]'%(symbols_in, symbols_out, pred_out))
    save.save(sess, './ckpt/model.ckpt')



    
    
    




