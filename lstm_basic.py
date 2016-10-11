#!/usr/bin/env python3

# Starting to implement Cheng & Voigt

import numpy as np
import tensorflow as tf

### Hyperparameters

LEARNING_RATE = 0.1
EPOCHS = 20
INPUT_SIZE = 3
NUM_HIDDEN = 6

# Currently hard-coding the batch size to be 1
# This reduces the amount of reshaping that Tensorflow needs to do tensor contraction

### Inputs and outputs

# Embedding matrix
x = tf.placeholder(tf.float32, [None, INPUT_SIZE])  # num tokens this doc, input vec size
# Coreference matrix
y = tf.placeholder(tf.float32, [None, None])        # num mentions this doc, num mentions this doc
# Referring expression matrix
s = tf.placeholder(tf.float32, [None, None])        # num mentions this doc, num tokens this doc

### Define the model

# RNN
cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_HIDDEN, state_is_tuple=True)
broadcast_x = tf.expand_dims(x, 0)  # Set batch size to 1
broadcast_outputs, states = tf.nn.dynamic_rnn(cell, broadcast_x, dtype=tf.float32)
outputs = tf.squeeze(broadcast_outputs)  # Remove the batch size index (of size 1)

# Entity representations
entities = tf.matmul(s, outputs)
# Normalise
normed_entities = tf.nn.l2_normalize(entities, 1)
# Cosine similarity
dot_product = tf.matmul(normed_entities, tf.transpose(normed_entities))  # num mentions, num mentions
nonneg_sim = tf.nn.relu(dot_product)

# Square distance from correct coreference
cost = tf.nn.l2_loss(nonneg_sim - y)

# TODO: currently the importance of a document grows quadratically with the number of referring expressions

# TODO: coreference evaluation metrics, after thresholding or clustering

# Train the model
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

init = tf.initialize_all_variables()  # Must be done AFTER introducing the optimizer (see http://stackoverflow.com/questions/33788989/tensorflow-using-adam-optimizer)

### Run the model

with tf.Session() as sess:
    # Dummy data
    my_x = [[[1.0, 2.3, 3.5], [1.0, 2.3, 3.5]], [[0.5, 0.42, 0.68], [0.5, 0.42, 0.68], [1.0, 2.3, 3.5], [1.0, -2.3, -3.5], [1.0, 2.3, 1.5]]]
    my_y = [[[1, 0], [0, 1]], [[1, 1, 0], [1, 1, 0], [0, 0, 1]]]
    my_s = [[[1, 0], [0, 1]], [[0.5, 0.5, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1]]]

    sess.run(init)
    print("Starting session")
    for step in range(EPOCHS):
        # TODO: take out hard coding of 2 documents
        for i in range(len(my_x)):
            current_dict = {x: my_x[i], y: my_y[i], s: my_s[i]}
            sess.run(optimizer, feed_dict=current_dict)
            loss = sess.run(cost, feed_dict=current_dict)
            coref_mat = sess.run(nonneg_sim, feed_dict=current_dict)
            # TODO include coreference evaluation metric
            # print("Epoch {}\nMinibatch loss {:.6f}\nTraining acc {:.5f}".format(step+1, loss, acc))
            print("Epoch {}\nMinibatch loss {:.6f}".format(step+1, loss))
            print(coref_mat)
