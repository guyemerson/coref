#!/usr/bin/env python

# Starting to implement Cheng & Voigt

# Uses MNIST data as input, which is nonsense, but an easy source of data

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 128
INPUT_SIZE = 49
NUM_STEPS = 16
NUM_HIDDEN = 100
NUM_CLASSES = 10

def my_rnn(x, weights, biases):
    """
    Build an RNN with a weight matrix applied to the outputs
    :param x: input tensor with shape [BATCH_SIZE, NUM_STEPS, INPUT_SIZE]
    :param weights: weight matrix with shape [NUM_HIDDEN, NUM_CLASSES]
    :param biases: bias weight matrix with shape [NUM_STEPS, NUM_CLASSES]
    """
    cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_HIDDEN, state_is_tuple=True)

    outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32) 
    # outputs is a tensor with shape [BATCH_SIZE, NUM_STEPS, NUM_HIDDEN]
    
    # Tensorflow doesn't currently support tensor contraction (I mean, seriously...)
    # So we have to convert to matrices and then convert back
    matrix_outputs = tf.reshape(outputs, [-1, NUM_HIDDEN])
    matrix_predictions = tf.matmul(matrix_outputs, W)
    predictions = tf.reshape(matrix_predictions, [BATCH_SIZE, NUM_STEPS, NUM_CLASSES])
    
    # Tensorflow *does* support broadcasting
    predictions += b
    
    return predictions

x = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_STEPS, INPUT_SIZE])
y = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_CLASSES])

W = tf.Variable(tf.zeros([NUM_HIDDEN, NUM_CLASSES]))
b = tf.Variable(tf.zeros([NUM_STEPS, NUM_CLASSES]))

pred = my_rnn(x, W, b)

# We're using a single MNIST image as a "sequence", so we need to broadcast the classes
broad_y = tf.tile(tf.reshape(y, [BATCH_SIZE, 1, NUM_CLASSES]), [1, NUM_STEPS, 1])

# To calculate the softmax cross-entropy, Tensorflow expects matrices,
# where each row is a distribution over classes
matrix_pred = tf.reshape(pred, [-1, NUM_CLASSES])
matrix_y = tf.reshape(broad_y, [-1, NUM_CLASSES])
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(matrix_pred, matrix_y)
# Find the average
cost = tf.reduce_mean(cross_entropy)

# Also caculate 1-best prediction accuracy
index_pred = tf.argmax(matrix_pred, 1)
index_y = tf.argmax(matrix_y, 1)
correct = tf.equal(index_pred, index_y)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Train the model
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

init = tf.initialize_all_variables()  # Must be done AFTER introducing the optimizer (see http://stackoverflow.com/questions/33788989/tensorflow-using-adam-optimizer)

with tf.Session() as sess:

    sess.run(init)
    for step in range(EPOCHS):
        batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
        batch_x = batch_x.reshape((BATCH_SIZE, NUM_STEPS, INPUT_SIZE))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        acc = sess.run(accuracy, feed_dict = {x: batch_x, y: batch_y})
        loss = sess.run(cost, feed_dict = {x: batch_x, y: batch_y})
        print("Epoch {}\nMinibatch loss {:.6f}\nTraining acc {:.5f}".format(step+1, loss, acc))
