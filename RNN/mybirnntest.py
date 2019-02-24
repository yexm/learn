# -- encoding:utf-8 --
"""
Create by yexm on 2019/2/23
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../mnist_data", one_hot=True)
learning_rate = 0.001
batch_size = 128
training_iters = 500
display_step =10

n_input = 28
n_step = 28
n_class = 10
n_hidden = 128

x = tf.placeholder("float32",[None,n_step,n_input])
y = tf.placeholder("float32",[None,n_class])
#双向lstm因为比单向的lstm多了反向cell，所以隐藏的输出维度是以前的2倍
weights={
    'out':tf.Variable(tf.random_normal([2*n_hidden,n_class]))
    }
biase = {
    'out':tf.Variable(tf.random_normal([n_class]))
}
def biRNN(x,weights,biase):
    x = tf.transpose(x,[1,0,2])
    x = tf.reshape(x,[-1,n_input])
    x = tf.split(x,n_step,0)

    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden)
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden)

    outs,_,_ = rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,x,dtype = tf.float32)

    return tf.matmul(outs[-1],weights['out'])+biase['out']
pred = biRNN(x,weights,biase)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    epoch = 1
    while epoch<training_iters:
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        batch_x= batch_x.reshape((batch_size, n_step, n_input))#要保证x和batch_x的shape是一样的
        sess.run(optimizer,feed_dict={x: batch_x, y: batch_y})
        if epoch % display_step  == 0:
            acc = sess.run(accuracy,feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost,feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(epoch * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        epoch += 1
    print("Optimization Finished!")

        # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_step, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
    sess.run(accuracy, feed_dict={x: test_data, y: test_label}))