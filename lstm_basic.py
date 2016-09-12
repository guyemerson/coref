#!/usr/bin/env python3

# Starting to implement Cheng & Voigt

import numpy as np
import tensorflow as tf

LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 1
INPUT_SIZE = 3
NUM_HIDDEN = 6
# NUM_STEPS = 16
# NUM_CLASSES = 10

# Dummy data
my_x = ([[[1.0, 2.3, 3.5], [1.0, 2.3, 3.5]]], [[[0.5, 0.42, 0.68], [0.5, 0.42, 0.68], [1.0, 2.3, 3.5]]])
my_y = ([[[0, 0, 1], [1, 0, 0], [0, 1, 0]]], [[[0, 0, 1], [1, 0, 0], [0, 1, 0]]])

def my_rnn(x, weights, biases):
    """
    Build an RNN with a weight matrix applied to the outputs
    :param x: input tensor with shape [BATCH_SIZE, num words in the doc, INPUT_SIZE]
    :param weights: weight matrix with shape [NUM_HIDDEN, NUM_HIDDEN]
    :param biases: bias weight matrix with shape [NUM_HIDDEN]
    """
    cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_HIDDEN, state_is_tuple=True)

    outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32) 
    # outputs is a tensor with shape [BATCH_SIZE, NUM_HIDDEN] (?)
    
    # TODO: probably some of this reshaping is no longer necessary
    # Tensorflow doesn't currently support tensor contraction (I mean, seriously...)
    # So we have to convert to matrices and then convert back
    matrix_outputs = tf.reshape(outputs, [-1, NUM_HIDDEN])
    matrix_predictions = tf.matmul(matrix_outputs, W)
    #    predictions = tf.reshape(matrix_predictions, [BATCH_SIZE, NUM_STEPS, NUM_CLASSES])
    predictions = matrix_predictions
    
    # Tensorflow *does* support broadcasting
    predictions += b
    
    return predictions

x = tf.placeholder(tf.float32, [BATCH_SIZE, None, INPUT_SIZE]) # batch size, num words this doc, input vec size
y = tf.placeholder(tf.float32, [BATCH_SIZE, None, None])       # batch size, num mentions this doc, num mentions this doc

W = tf.Variable(tf.zeros([NUM_HIDDEN, NUM_HIDDEN]))
b = tf.Variable(tf.zeros([NUM_HIDDEN]))

# TODO: use S in my_rnn
S = tf.placeholder(tf.float32, [BATCH_SIZE, None, None])   # batch size, num mentions this doc, num words this doc

pred = my_rnn(x, W, b)

# # To calculate the softmax cross-entropy, Tensorflow expects matrices,
# # where each row is a distribution over classes
# matrix_pred = tf.reshape(pred, [-1, NUM_CLASSES])
# matrix_y = tf.reshape(broad_y, [-1, NUM_CLASSES])
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(matrix_pred, matrix_y)
# # Find the average
# cost = tf.reduce_mean(cross_entropy)


# TODO: replace dummy cross entropy calculation with real one
pointwise_cross_entropy = tf.reduce_sum(pred)
cost = tf.reduce_mean(pointwise_cross_entropy)

# TODO: this is now broken: fix
# # Also caculate 1-best prediction accuracy
# index_pred = tf.argmax(matrix_pred, 1)
# index_y = tf.argmax(matrix_y, 1)
# correct = tf.equal(index_pred, index_y)
# accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Train the model
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

init = tf.initialize_all_variables()  # Must be done AFTER introducing the optimizer (see http://stackoverflow.com/questions/33788989/tensorflow-using-adam-optimizer)

with tf.Session() as sess:

    sess.run(init)
    print("Starting session")
    for step in range(EPOCHS):
        # TODO: take out hard coding of 2 documents
        for i in range(2):
            current_x = my_x[i]
            current_y = my_y[i]
            sess.run(optimizer, feed_dict={x: current_x, y: current_y})
            # acc = sess.run(accuracy, feed_dict = {x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict = {x: current_x, y: current_y})
            # print("Epoch {}\nMinibatch loss {:.6f}\nTraining acc {:.5f}".format(step+1, loss, acc))
            print("Epoch {}\nMinibatch loss {:.6f}".format(step+1, loss))
