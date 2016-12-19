#!/usr/bin/env python3

# Starting to implement Cheng & Voigt

import numpy as np
import tensorflow as tf

from conll import ConllCorpusReader
from matrix_gen import get_s_matrix, coref_matrix
corpus_dir = "/anfs/bigdisc/kh562/Corpora/conll-2011/"
conll_reader = ConllCorpusReader(corpus_dir).parse_corpus()

tf.set_random_seed(1234)


### Hyperparameters

LEARNING_RATE = 0.1
EPOCHS = 20
# TODO: input size to be set to 300 when using cached vectors (i.e. real data)
INPUT_SIZE = 300
NUM_HIDDEN = 600

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
cost = tf.nn.l2_loss(nonneg_sim - y) + 0.1*sum([tf.reduce_sum(x**2) for x in tf.trainable_variables()])
error_rate = tf.truediv(cost, tf.to_float((tf.shape(y)[0] * (tf.shape(y)[0] - 1))))

# TODO: currently the importance of a document grows quadratically with the number of referring expressions

# TODO: coreference evaluation metrics, after thresholding or clustering

# Train the model
adam = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
optimizer = adam.minimize(cost)

init = tf.initialize_all_variables()  # Must be done AFTER introducing the optimizer (see http://stackoverflow.com/questions/33788989/tensorflow-using-adam-optimizer)

### Run the model

with tf.Session() as sess:
    # code to load the cached document vectors
    train_docs = np.load("/anfs/bigdisc/kh562/Corpora/conll-2011/training_docs.npz", encoding='latin1')["matrices"]
    test_docs = np.load("/anfs/bigdisc/kh562/Corpora/conll-2011/test_docs.npz", encoding='latin1')["matrices"]

    train_conll_docs = conll_reader.get_conll_docs("train")
    s_matrix = [get_s_matrix(x) for x in train_conll_docs]
    train_coref_matrix = [coref_matrix(x) for x in train_conll_docs]
#    nonzero, = s_matrix[0].nonzero()
#    print(nonzero)
#    print(train_conll_docs[0].get_document_tokens()[nonzero.min():nonzero.max()+1])

    test_conll_docs = conll_reader.get_conll_docs("test")
    test_s_matrix = [get_s_matrix(x) for x in test_conll_docs]
    test_coref_matrix = [coref_matrix(x) for x in test_conll_docs]

    varnames = [v.name for v in tf.trainable_variables()]
    print(varnames)

#   RNN/BasicLSTMCell/Linear/Matrix:0, RNN/BasicLSTMCell/Linear/Bias:0

    maxweight = [tf.reduce_max(x) for x in tf.trainable_variables()]
    meanweight = [tf.reduce_mean(x) for x in tf.trainable_variables()]
    gradients = [tf.reduce_max(x) for x, _ in adam.compute_gradients(cost)]
    meangradients = [tf.reduce_mean(x) for x, _ in adam.compute_gradients(cost)]

    sess.run(init)
    print("Starting session")
    for step in range(EPOCHS):
        for i in range(len(train_conll_docs)):
            current_dict = {x: train_docs[i], y: train_coref_matrix[i], s: s_matrix[i]}
            sess.run(optimizer, feed_dict=current_dict)
            loss = sess.run(error_rate, feed_dict=current_dict)
#            coref_mat = sess.run(nonneg_sim, feed_dict=current_dict)
            # TODO include coreference evaluation metric
            # print("Epoch {}\nMinibatch loss {:.6f}\nTraining acc {:.5f}".format(step+1, loss, acc))
            print("Epoch {}\nDocument {}\nMinibatch loss {:.6f}".format(step+1, i, loss))
#            print(coref_mat)
            current_mean_weight = sess.run(meanweight)
            print("Mean weight", current_mean_weight)
            current_max_weight = sess.run(maxweight)
            print("Max weight", current_max_weight)
            current_mean_gradient = sess.run(meangradients, feed_dict=current_dict)
            print("Mean gradient", current_mean_gradient)
            current_max_gradient = sess.run(gradients, feed_dict=current_dict)
            print("Max gradient", current_max_gradient)

        for i in range(len(test_docs)):
            current_dict = {x: test_docs[i], y: test_coref_matrix[i], s: test_s_matrix[i]}
            loss = sess.run(error_rate, feed_dict=current_dict)
            print("Document {}\nLoss {:.6f}".format(i, loss))
            
