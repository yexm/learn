#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib as contrib
tf.set_random_seed(1)

##导入数据
mnist = input_data.read_data_sets('../mnist_data', one_hot=True)

##超参数
lr = 0.001  # learning rate
training_iters = 100000  # train step 上限
batch_size = 128
n_inputs = 28  # MNIST data input (img shape: 28*28)
n_steps = 28  # time steps
n_hidden_units = 128  # neurons in hidden layer
n_classes = 10  # MNIST classes (0-9 digits)

## x,y placeholder
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

## 对 weights biases 的初始值的定义
weights = {
    # shape (28, 128)
    "in": tf.Variable(tf.truncated_normal([n_inputs, n_hidden_units])),
    # shape (128, 10)
    'out': tf.Variable(tf.truncated_normal([n_hidden_units, n_classes]))
}
biases = {
    # shape (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # shape (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

# 定义 RNN 的主体结构（input_layer, cell, output_layer）
def RNN(X, weights, biases):
    # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法
    # X ==> (128 batches * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])
    # X_in = W*X + b
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batches, 28 steps, 128 hidden) 换回3维
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # 使用 basic LSTM Cell.
    lstm_cell = contrib.rnn.BasicLSTMCell(num_units=n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    ## 如果 inputs 为 (batches, steps, inputs) ==> time_major=False; 如果 inputs 为 (steps, batches, inputs) ==> time_major=True;
    outputs, final_state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=X_in, initial_state=init_state,
                                             time_major=False)

    ###最后是 output_layer 和 return 的值. 因为这个例子的特殊性, 有两种方法可以求得 results.
    ### 方法一： 直接调用final_state 中的 h_state (final_state[1]) 来进行运算
    results = tf.matmul(final_state[1], weights['out']) + biases['out']
    ## 调用最后一个 outputs (在这个例子中,和上面的final_state[1]是一样的)
    # 把 outputs 变成 列表 [(batch, outputs)..] * steps
    # outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    # results = tf.matmul(outputs[-1], weights['out']) + biases['out']  # 选取最后一个 output
    return results

# 定义好了 RNN 主体结构后, 我们就可以来计算 cost 和 train_op
pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
# train_op = tf.train.AdadeltaOptimizer(lr).minimize(cost)  ## 这出错了啦！！！！！！

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={x: batch_xs, y: batch_ys})

        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys}))
        step += 1
