#!/usr/bin/env python3

# Starting to implement Cheng & Voigt

import numpy as np
import tensorflow as tf
import argparse
from collections import defaultdict

from conll import ConllCorpusReader
from matrix_gen import get_s_matrix, coref_matrix
from evaluation import get_evaluation

parser = argparse.ArgumentParser()
parser.add_argument("--print_document_scores", help="Print metrics per document during training", action="store_true")
parser.add_argument("--print_minibatch_loss", help="Print minibatch (document) loss during training", action="store_true")
parser.add_argument("--print_dev_loss", help="Print minibatch (document) loss during evaluation on the dev set", action="store_true")
parser.add_argument("--epochs", help="Number of training epochs", default=20)
parser.add_argument("--learning_rate", help="Learning rate for training", default=0.1)
parser.add_argument("--hidden_size", help="Number of hidden units", default=600)
parser.add_argument("--threshold", help="Threshold value for coference (between 0 and 1)", default=0.79)
parser.add_argument("--reg_weight", help="The weight of regularization function", default=0.1)
parser.add_argument("--print_coref_matrices", help="Print gold and predicted coreference matrices", action="store_true")
args = parser.parse_args()

corpus_dir = "/anfs/bigdisc/kh562/Corpora/conll-2012/v4/data/"
conll_reader = ConllCorpusReader(corpus_dir).parse_corpus()

metrics = ['muc', 'bcub', 'ceafm', 'ceafe', 'blanc', 'lea']

tf.set_random_seed(1234)

### Hyperparameters

LEARNING_RATE = float(args.learning_rate)
EPOCHS = int(args.epochs)
# TODO: input size to be set to 300 when using cached vectors (i.e. real data)
INPUT_SIZE = 300
NUM_HIDDEN = int(args.hidden_size)
# threshold for cosine similarity on coref matrix (C) 
THRESHOLD=float(args.threshold)
REGULARIZATION_WEIGHT=float(args.reg_weight)

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

# OLD COST (square distance)
# cost = tf.nn.l2_loss(nonneg_sim - y)

# NEW COST (cross entropy)
cost =  - (1/tf.size(y)) * tf.reduce_sum(y*tf.log(nonneg_sim+(1e-10)) + (1-y)*tf.log(1+(1e-10)-nonneg_sim))
reg = REGULARIZATION_WEIGHT*sum([tf.reduce_sum(x**2) for x in tf.trainable_variables()])
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
    train_docs = np.load("/anfs/bigdisc/kh562/Corpora/conll-2012/training_docs.npz", encoding='latin1')["matrices"]
    dev_docs = np.load("/anfs/bigdisc/kh562/Corpora/conll-2012/development_docs.npz", encoding='latin1')["matrices"]

    train_conll_docs = conll_reader.get_conll_docs("train")
    train_s_matrix = [get_s_matrix(x) for x in train_conll_docs]
    train_coref_matrix = [coref_matrix(x) for x in train_conll_docs]
#    nonzero, = s_matrix[0].nonzero()
#    print(nonzero)
#    print(train_conll_docs[0].get_document_tokens()[nonzero.min():nonzero.max()+1])

    dev_conll_docs = conll_reader.get_conll_docs("development")
    dev_s_matrix = [get_s_matrix(x) for x in dev_conll_docs]
    dev_coref_matrix = [coref_matrix(x) for x in dev_conll_docs]

    sess.run(init)
    print("Starting session")
    for step in range(EPOCHS):
        metrics_this_epoch = defaultdict(lambda: defaultdict(list))
        losses_this_epoch = []
        for i in range(len(train_conll_docs)):
            current_dict = {x: train_docs[i], y: train_coref_matrix[i], s: train_s_matrix[i]}
            sess.run(optimizer, feed_dict=current_dict)

            loss = sess.run(error_rate, feed_dict=current_dict)
            coref_mat = sess.run(nonneg_sim, feed_dict=current_dict)
            # get evaluation of current predicted coref matrix
            losses_this_epoch.append(loss)
            evals = get_evaluation(train_conll_docs[i],coref_mat,THRESHOLD)
            if args.print_coref_matrices:
                print("GOLD COREFERENCE MATRIX")
                print(train_coref_matrix[i])
                print("PREDICTED COREFERENCE MATRIX")
                print(coref_mat)
            for m in metrics:
                metrics_this_epoch[m]['R'].append(evals[m][0])
                metrics_this_epoch[m]['P'].append(evals[m][1])
                metrics_this_epoch[m]['F1'].append(evals[m][2])
            metrics_this_epoch['conll']['avg'].append(evals['avg'])
            if args.print_document_scores:
                print(evals["formatted"])
            if args.print_minibatch_loss:
                # print("Epoch {}\nMinibatch loss {:.6f}\nTraining acc {:.5f}".format(step+1, loss, acc))
                print("Epoch {}\nDocument {}\nMinibatch loss {:.6f}".format(step+1, i, loss))


        avg_scores = defaultdict(lambda: defaultdict(float))
        for m in metrics_this_epoch:
            for t in metrics_this_epoch[m]:
                avg_scores[m][t] = sum(metrics_this_epoch[m][t]) / len(metrics_this_epoch[m][t])
        avg_loss=sum(losses_this_epoch) / len(losses_this_epoch)
        formatted = "\tR\tP\tF1\n"
        for m in metrics:
            formatted += m + "\t" + format(avg_scores[m]['R'], '.2f') + "\t" +  format(avg_scores[m]['P'], '.2f') + "\t" + format(avg_scores[m]['F1'], '.2f') + "\n"
        formatted += "\n"
        formatted += "conll\t\t\t" + format(avg_scores['conll']['avg'], '.2f') + "\n"
        print("AVERAGE METRICS FOR EPOCH", step+1)
        print(formatted)
        print("AVERAGE LOSS FOR EPOCH{}\n{:.6f}".format(step+1, avg_loss))            

        print("Evaluating on dev set\n")
        for i in range(len(dev_conll_docs)):
            current_dict = {x: dev_docs[i], y: dev_coref_matrix[i], s: dev_s_matrix[i]}
            loss = sess.run(error_rate, feed_dict=current_dict)
            if args.print_dev_loss:
                print("Document {}\nLoss {:.6f}".format(i, loss))
            coref_mat = sess.run(nonneg_sim, feed_dict=current_dict)
            losses_this_epoch.append(loss)
            # get evaluation of current predicted coref matrix
            evals = get_evaluation(dev_conll_docs[i],coref_mat,THRESHOLD)
            if args.print_coref_matrices:
                print("GOLD COREFERENCE MATRIX")
                print(dev_coref_matrix[i])
                print("PREDICTED COREFERENCE MATRIX")
                print(coref_mat)
            for m in metrics:
                metrics_this_epoch[m]['R'].append(evals[m][0])
                metrics_this_epoch[m]['P'].append(evals[m][1])
                metrics_this_epoch[m]['F1'].append(evals[m][2])
            metrics_this_epoch['conll']['avg'].append(evals['avg'])
        avg_scores = defaultdict(lambda: defaultdict(float))
        for m in metrics_this_epoch:
            for t in metrics_this_epoch[m]:
                avg_scores[m][t] = sum(metrics_this_epoch[m][t]) / len(metrics_this_epoch[m][t])
        avg_loss=sum(losses_this_epoch) / len(losses_this_epoch)
        formatted = "\tR\tP\tF1\n"
        for m in metrics:
            formatted += m + "\t" + format(avg_scores[m]['R'], '.2f') + "\t" +  format(avg_scores[m]['P'], '.2f') + "\t" + format(avg_scores[m]['F1'], '.2f') + "\n"
        formatted += "\n"
        formatted += "conll\t\t\t" + format(avg_scores['conll']['avg'], '.2f') + "\n"
        print("AVERAGE METRICS ON DEV SET, EPOCH", step+1)
        print(formatted)
        print("AVERAGE LOSS FOR EPOCH{}\n{:.6f}".format(step+1, avg_loss))     
