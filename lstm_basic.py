#!/usr/bin/env python3

# Starting to implement Cheng & Voigt

import numpy as np
import tensorflow as tf

from conll import ConllCorpusReader
from matrix_gen import get_s_matrix, coref_matrix
from evaluation import get_evaluation

corpus_dir = "/anfs/bigdisc/kh562/Corpora/conll-2011/"
conll_reader = ConllCorpusReader(corpus_dir).parse_corpus()

tf.set_random_seed(1234)

### Hyperparameters

LEARNING_RATE = 0.1
EPOCHS = 20
# TODO: input size to be set to 300 when using cached vectors (i.e. real data)
INPUT_SIZE = 300
NUM_HIDDEN = 600
# threshold for cosine similarity on coref matrix (C) 
THRESHOLD=0.79

# Currently hard-coding the batch size to be 1
# This reduces the amount of reshaping that Tensorflow needs to do tensor contraction

### Inputs and outputs

# Embedding matrix
x = tf.placeholder(tf.float64, [None, INPUT_SIZE])  # num tokens this doc, input vec size
# Coreference matrix
y = tf.placeholder(tf.float64, [None, None])        # num mentions this doc, num mentions this doc
# Referring expression matrix
s = tf.placeholder(tf.float64, [None, None])        # num mentions this doc, num tokens this doc

### Define the model

# RNN
cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(NUM_HIDDEN, state_is_tuple=True)
broadcast_x = tf.expand_dims(x, 0)  # Set batch size to 1
broadcast_outputs, states = tf.nn.dynamic_rnn(cell, broadcast_x, dtype=tf.float64)
outputs = tf.squeeze(broadcast_outputs)  # Remove the batch size index (of size 1)

# Entity representations
entities = tf.matmul(s, outputs)
# Normalise
normed_entities = tf.nn.l2_normalize(entities, 1)
# Cosine similarity
dot_product = tf.matmul(normed_entities, tf.transpose(normed_entities))  # num mentions, num mentions
nonneg_sim = tf.nn.relu(dot_product)

# Square distance from correct coreference
# cost = tf.nn.l2_loss(nonneg_sim - y)


# OLD COST
# cost = tf.nn.l2_loss(nonneg_sim - y)
cost =  - (1/tf.shape(y)[0]**2) * tf.reduce_sum(y*tf.log(nonneg_sim+(1e-10)) + (1-y)*tf.log(1+(1e-10)-nonneg_sim))
reg = 0.1*sum([tf.reduce_sum(x**2) for x in tf.trainable_variables()])
cost = cost + reg


error_rate = tf.truediv(cost, tf.cast((tf.shape(y)[0] * (tf.shape(y)[0] - 1)), tf.float64))

# TODO: currently the importance of a document grows quadratically with the number of referring expressions

# TODO: coreference evaluation metrics, after thresholding or clustering

# Train the model
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

init = tf.initialize_all_variables()  # Must be done AFTER introducing the optimizer (see http://stackoverflow.com/questions/33788989/tensorflow-using-adam-optimizer)

### Run the model

with tf.Session() as sess:
    # code to load the cached document vectors
    train_docs = np.load("/anfs/bigdisc/kh562/Corpora/conll-2011/training_docs.npz", encoding='latin1')["matrices"]
    test_docs = np.load("/anfs/bigdisc/kh562/Corpora/conll-2011/test_docs.npz", encoding='latin1')["matrices"]

    train_conll_docs = conll_reader.get_conll_docs("train")
    train_s_matrix = [get_s_matrix(x) for x in train_conll_docs]
    train_coref_matrix = [coref_matrix(x) for x in train_conll_docs]
#    nonzero, = s_matrix[0].nonzero()
#    print(nonzero)
#    print(train_conll_docs[0].get_document_tokens()[nonzero.min():nonzero.max()+1])

    test_conll_docs = conll_reader.get_conll_docs("test")
    test_s_matrix = [get_s_matrix(x) for x in test_conll_docs]
    test_coref_matrix = [coref_matrix(x) for x in test_conll_docs]

    sess.run(init)
    print("Starting session")
    for step in range(EPOCHS):
        for i in range(len(train_conll_docs)):
            current_dict = {x: train_docs[i], y: train_coref_matrix[i], s: train_s_matrix[i]}
            sess.run(optimizer, feed_dict=current_dict)

            loss = sess.run(error_rate, feed_dict=current_dict)
            coref_mat = sess.run(nonneg_sim, feed_dict=current_dict)
            # get evaluation of current predicted coref matrix
            print(get_evaluation(train_conll_docs[i],coref_mat,THRESHOLD)["formatted"])
#            print("Epoch {}\nMinibatch loss {:.6f}\nTraining acc {:.5f}".format(step+1, loss, acc))
            print("Epoch {}\nDocument {}\nMinibatch loss {:.6f}".format(step+1, i, loss))

        for i in range(len(test_docs)):
            current_dict = {x: test_docs[i], y: test_coref_matrix[i], s: test_s_matrix[i]}
            loss = sess.run(error_rate, feed_dict=current_dict)
            print("Document {}\nLoss {:.6f}".format(i, loss))
            
