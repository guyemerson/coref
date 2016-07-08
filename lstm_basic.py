#!/usr/bin/env python

# Starting to implement Cheng & Voigt

# Uses MNIST data as input, which is nonsense, but an easy source of data

import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
epochs = 10
batch_size = 128
input_size = 28
num_steps = 28
num_hidden = 128 
num_classes = 10 

def my_rnn(x, weights, biases):   # an RNN cell with a weight matrix applied to the outputs

    x = tf.transpose(x, [1, 0, 2])   # [batch_size, num_steps, input_size] => [num_steps, batch_size, input_size]
    x = tf.reshape(x, [-1, input_size])   # [num_steps * batch_size, input_size]
    x = tf.split(0, num_steps, x)    # list of num_steps tensors, each of shape [batch_size, input_size]

    cell = rnn_cell.BasicLSTMCell(num_hidden)

    outputs, states = rnn.rnn(cell, x, dtype=tf.float32)  # outputs = list of num_steps tensors, each of shape [batch_size, num_hidden]

    outputs = tf.pack(outputs) # outputs = tensor of shape [num_steps, batch_size, num_hidden]
    outputs = tf.transpose(outputs, [1, 0, 2])  # outputs = tensor of shape [batch_size, num_steps, num_hidden]
    outputs = tf.split(0, batch_size, outputs)

    batch_out = []
    for z in outputs:
        batch_out.append(tf.matmul(tf.squeeze(z), weights) + biases)

    return tf.pack(batch_out)     # [batch_size, num_steps, num_classes]


x = tf.placeholder(tf.float32, [batch_size, num_steps, input_size])
y = tf.placeholder(tf.float32, [batch_size, num_classes])

W = tf.Variable(tf.zeros([num_hidden, num_classes]))
b = tf.Variable(tf.zeros([num_steps, num_classes]))
    
pred = my_rnn(x, W, b)   # [batch_size, num_steps, num_classes]
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.reshape(pred, [batch_size, num_steps * num_classes]),   tf.tile(y, [1,num_steps])     ))  / batch_size    # not 100% sure the correct dimensions are getting matched up here


correct_pred = tf.equal(tf.argmax(tf.reshape(pred, [batch_size, num_steps * num_classes]), 1), tf.argmax(tf.tile(y, [1,14]), 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

init = tf.initialize_all_variables()  # Must be done AFTER introducing the optimizer (see http://stackoverflow.com/questions/33788989/tensorflow-using-adam-optimizer)

with tf.Session() as sess:

    sess.run(init)
    step = 1
    while step < epochs:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, num_steps, input_size))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        acc = sess.run(accuracy, feed_dict = {x: batch_x, y: batch_y})
        loss = sess.run(cost, feed_dict = {x: batch_x, y: batch_y})
        print "Epoch", str(step * batch_size), "Minibatch loss", "{:.6f}".format(loss), "Training acc", "{:.5f}".format(acc)
        step += 1


