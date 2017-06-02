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
parser.add_argument("--threshold", help="Threshold value for coference (between 0 and 1)", default=0.79, type=float)
parser.add_argument("--reg_weight", help="The weight of regularization function", default=0.1, type=float)
parser.add_argument("--print_coref_matrices", help="Print gold and predicted coreference matrices", action="store_true")
parser.add_argument("--additional_features",help="Use vectors containing additional features",action="store_true")
parser.add_argument("--model_dir", help="Directory for saving models", default="models")
parser.add_argument("--eval_on_model", help="Path to the model", default="none")
args = parser.parse_args()

corpus_dir = "/anfs/bigdisc/kh562/Corpora/conll-2012/"
conll_reader = ConllCorpusReader(corpus_dir).parse_corpus()

metrics = ['muc', 'bcub', 'ceafm', 'ceafe', 'blanc', 'lea']

tf.set_random_seed(99)

### Hyperparameters

LEARNING_RATE = float(args.learning_rate)
EPOCHS = int(args.epochs)
INPUT_SIZE = 362 if args.additional_features else 300
NUM_HIDDEN = int(args.hidden_size)
# threshold for cosine similarity on coref matrix (C) 
THRESHOLD = args.threshold
REGULARIZATION_WEIGHT = args.reg_weight

# Currently hard-coding the batch size to be 1
# This reduces the amount of reshaping that Tensorflow needs to do tensor contraction

### Inputs and outputs

# Embedding matrix
x = tf.placeholder(tf.float32, [None, INPUT_SIZE])  # num tokens this doc, input vec size
tf.add_to_collection('x', x)
# Coreference matrix
y = tf.placeholder(tf.float32, [None, None])        # num mentions this doc, num mentions this doc
tf.add_to_collection('y', y)
# Referring expression matrix
s = tf.placeholder(tf.float32, [None, None])        # num mentions this doc, num tokens this doc
tf.add_to_collection('s', s)

### Define the model

# RNN
cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(NUM_HIDDEN, state_is_tuple=True)
broadcast_x = tf.expand_dims(x, 0, name="op_broadcast_x")  # Set batch size to 1
broadcast_outputs, states = tf.nn.dynamic_rnn(cell, broadcast_x, dtype=tf.float32)
outputs = tf.squeeze(broadcast_outputs, name="op_outputs")  # Remove the batch size index (of size 1)

# Entity representations
entities = tf.matmul(s, outputs, name="op_entities")
# Normalise
normed_entities = tf.nn.l2_normalize(entities, 1, name="op_normed_entities")
# Cosine similarity
dot_product = tf.matmul(normed_entities, tf.transpose(normed_entities), name="op_dot_product")  # num mentions, num mentions
nonneg_sim = tf.nn.relu(dot_product, name="op_nonneg_sim")

cost =  tf.truediv(-tf.reduce_sum(y*tf.log(nonneg_sim+(1e-5)) + (1-y)*tf.log(1+(1e-5)-nonneg_sim)), tf.cast(tf.size(y), tf.float32), name="op_cost")
reg = tf.multiply(REGULARIZATION_WEIGHT, sum([tf.reduce_sum(x**2) for x in tf.trainable_variables()]), name="op_reg")
regcost = tf.add(cost, reg, name="op_regcost")

# cost1 =  y*tf.log(nonneg_sim+(1e-5))
# cost2 =  (1-y)*tf.log(1+(1e-5)-nonneg_sim)
# cost3 =  - tf.reduce_sum(y*tf.log(nonneg_sim+(1e-5)) + (1-y)*tf.log(1+(1e-5)-nonneg_sim))
# cost4 =  tf.cast(tf.size(y), tf.float32)
# cost5 =  tf.truediv(cost3, cost4)

# TODO: currently the importance of a document grows quadratically with the number of referring expressions

# TODO: coreference evaluation metrics, after thresholding or clustering

# Train the model
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

init = tf.global_variables_initializer()

### Run the model

# code to load the cached document vectors
if args.additional_features:
    train_docs = np.load("/anfs/bigdisc/kh562/Corpora/conll-2012/training_docs_new.npz", encoding='latin1')["matrices"]
    dev_docs = np.load("/anfs/bigdisc/kh562/Corpora/conll-2012/development_docs_new.npz", encoding='latin1')["matrices"]
    test_docs = np.load("/anfs/bigdisc/kh562/Corpora/conll-2012/test_docs_new.npz", encoding='latin1')["matrices"]
else:
    train_docs = np.load("/anfs/bigdisc/kh562/Corpora/conll-2012/training_docs.npz", encoding='latin1')["matrices"]
    dev_docs = np.load("/anfs/bigdisc/kh562/Corpora/conll-2012/development_docs.npz", encoding='latin1')["matrices"]
    test_docs = np.load("/anfs/bigdisc/kh562/Corpora/conll-2012/test_docs.npz", encoding='latin1')["matrices"]

train_conll_docs = conll_reader.get_conll_docs("train")
train_s_matrix = [get_s_matrix(x) for x in train_conll_docs]
train_coref_matrix = [coref_matrix(x) for x in train_conll_docs]
#    nonzero, = s_matrix[0].nonzero()
#    print(nonzero)
#    print(train_conll_docs[0].get_document_tokens()[nonzero.min():nonzero.max()+1])

dev_conll_docs = conll_reader.get_conll_docs("development")
dev_s_matrix = [get_s_matrix(x) for x in dev_conll_docs]
dev_coref_matrix = [coref_matrix(x) for x in dev_conll_docs]

saver = tf.train.Saver()

if args.eval_on_model == 'none':
    with tf.Session() as sess:
        sess.run(init)
        print("Starting session")
        for step in range(EPOCHS):
            metrics_this_epoch = defaultdict(lambda: defaultdict(list))
            losses_this_epoch = []
            skipped = 0
            for i in range(len(train_conll_docs)):
                current_dict = {x: train_docs[i], y: train_coref_matrix[i], s: train_s_matrix[i]}
                if train_coref_matrix[i].size == 0:
                    skipped += 1
                    continue
                coref_mat, loss, _ = sess.run([nonneg_sim, cost, optimizer], feed_dict=current_dict)
                print("coref_mat = nonneg_sim", coref_mat)
                print("loss", loss)
                if args.print_minibatch_loss:
                    print("Epoch {}\nDocument {}\nMinibatch loss {:.6f}".format(step+1, i+1, loss))
		# get evaluation of current predicted coref matrix
                losses_this_epoch.append(loss)
                try:   # avoid errors at some thresholds
                    evals = get_evaluation(train_conll_docs[i],coref_mat,THRESHOLD)
                except:
                    continue
                if args.print_coref_matrices:
                    print("GOLD COREFERENCE MATRIX")
                    print(train_coref_matrix[i])
                    print("PREDICTED COREFERENCE MATRIX")
                    print(coref_mat)
                for m in metrics:
                    metrics_this_epoch[m]['R'].append(evals[m][0])
                    metrics_this_epoch[m]['P'].append(evals[m][1])
                metrics_this_epoch['conll']['avg'].append(evals['avg'])
                if args.print_document_scores:
                    print(evals["formatted"])
            avg_scores = defaultdict(lambda: defaultdict(float))
            for m in metrics_this_epoch:
                for t in metrics_this_epoch[m]:
                    avg_scores[m][t] = sum(metrics_this_epoch[m][t]) / len(metrics_this_epoch[m][t])
            avg_loss=sum(losses_this_epoch) / len(losses_this_epoch)
            formatted = "\tR\tP\tF1\n"
            for m in metrics:
                formatted += m + "\t" + format(avg_scores[m]['R'], '.2f') + "\t" +  format(avg_scores[m]['P'], '.2f') + "\t" + format((2*avg_scores[m]['P']*avg_scores[m]['R'])/(avg_scores[m]['P'] + avg_scores[m]['R']), '.2f') + "\n"
            formatted += "\n"
            formatted += "conll\t\t\t" + format(avg_scores['conll']['avg'], '.2f') + "\n"
            print("AVERAGE METRICS FOR EPOCH", step+1)
            print(formatted)
            print("AVERAGE LOSS FOR EPOCH {}\n{:.6f}".format(step+1, avg_loss))
            print("Skipped {} documents with zero mentions".format(skipped))

            print("Saving model")
            saver.save(sess, args.model_dir + '/' + 'model_' + str(LEARNING_RATE) + '_' + str(EPOCHS) + '_' + str(NUM_HIDDEN) + '_' + str(REGULARIZATION_WEIGHT) + '_epoch' + str(step+1))

            print("Evaluating on dev set\n")    
            for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                print("EPOCH", i+1, "THRESHOLD", thresh)
                skipped = 0
                for i in range(len(dev_conll_docs)):
                    current_dict = {x: dev_docs[i], y: dev_coref_matrix[i], s: dev_s_matrix[i]}
                    if dev_coref_matrix[i].size == 0:
                        skipped += 1
                        continue
                loss = sess.run(cost, feed_dict=current_dict)
                if args.print_dev_loss:
                    print("Document {} Loss {:.6f}".format(i, loss))
                coref_mat = sess.run(nonneg_sim, feed_dict=current_dict)
                losses_this_epoch.append(loss)
                # get evaluation of current predicted coref matrix
                try:   # avoid errors at some thresholds
                    evals = get_evaluation(dev_conll_docs[i],coref_mat,thresh)
                except:
                    continue
                if args.print_coref_matrices:
                    print("GOLD COREFERENCE MATRIX")
                    print(dev_coref_matrix[i])
                    print("PREDICTED COREFERENCE MATRIX")
                    print(coref_mat)
                for m in metrics:
                    metrics_this_epoch[m]['R'].append(evals[m][0])
                    metrics_this_epoch[m]['P'].append(evals[m][1])
                metrics_this_epoch['conll']['avg'].append(evals['avg'])
                avg_scores = defaultdict(lambda: defaultdict(float))
                for m in metrics_this_epoch:
                    for t in metrics_this_epoch[m]:
                        avg_scores[m][t] = sum(metrics_this_epoch[m][t]) / len(metrics_this_epoch[m][t])
                avg_loss=sum(losses_this_epoch) / len(losses_this_epoch)
                formatted = "\tR\tP\tF1\n"
                for m in metrics:
                    formatted += m + "\t" + format(avg_scores[m]['R'], '.2f') + "\t" +  format(avg_scores[m]['P'], '.2f') + "\t" + format((2*avg_scores[m]['P']*avg_scores[m]['R'])/(avg_scores[m]['P'] + avg_scores[m]['R']), '.2f') + "\n"
                formatted += "\n"
                formatted += "conll\t\t\t" + format(avg_scores['conll']['avg'], '.2f') + "\n"
                print("AVERAGE METRICS ON DEV SET EPOCH", step+1, "THRESHOLD", thresh)
                print(formatted)
                print("AVERAGE LOSS ON DEV SET\n{:.6f}".format(avg_loss))     
                print("Skipped {} documents with zero mentions".format(skipped))


with tf.Session() as sess:
    # Initialize variables
    sess.run(init)
    metrics_on_dev = defaultdict(lambda: defaultdict(list))
    losses_on_dev = []
    new_saver = tf.train.Saver()
    if(args.eval_on_model=='none'):
        new_saver.restore(sess, args.model_dir + '/' + 'model_' + str(LEARNING_RATE) + '_' + str(EPOCHS) + '_' + str(NUM_HIDDEN) + '_' + str(REGULARIZATION_WEIGHT) + '_epoch' + str(EPOCHS))
    else:
        new_saver.restore(sess, args.eval_on_model)
    print("Evaluating on dev set\n")
    skipped = 0
    for i in range(len(dev_conll_docs)):
        current_dict = {x: dev_docs[i], y: dev_coref_matrix[i], s: dev_s_matrix[i]}
        if dev_coref_matrix[i].size == 0:
            skipped += 1
            continue
        loss = sess.run(cost, feed_dict=current_dict)
        if args.print_dev_loss:
            print("Document {}\nLoss {:.6f}".format(i, loss))
        coref_mat = sess.run(nonneg_sim, feed_dict=current_dict)
        losses_on_dev.append(loss)
        # get evaluation of current predicted coref matrix
        evals = get_evaluation(dev_conll_docs[i],coref_mat,THRESHOLD)
        if args.print_coref_matrices:
            print("GOLD COREFERENCE MATRIX")
            print(dev_coref_matrix[i])
            print("PREDICTED COREFERENCE MATRIX")
            print(coref_mat)
        for m in metrics:
            metrics_on_dev[m]['R'].append(evals[m][0])
            metrics_on_dev[m]['P'].append(evals[m][1])
            metrics_on_dev[m]['F1'].append(evals[m][2])
        metrics_on_dev['conll']['avg'].append(evals['avg'])
    avg_scores = defaultdict(lambda: defaultdict(float))
    for m in metrics_on_dev:
        for t in metrics_on_dev[m]:
            avg_scores[m][t] = sum(metrics_on_dev[m][t]) / len(metrics_on_dev[m][t])
    avg_loss=sum(losses_on_dev) / len(losses_on_dev)
    formatted = "\tR\tP\tF1\n"
    for m in metrics:
        formatted += m + "\t" + format(avg_scores[m]['R'], '.2f') + "\t" +  format(avg_scores[m]['P'], '.2f') + "\t" + format(avg_scores[m]['F1'], '.2f') + "\n"
    formatted += "\n"
    formatted += "conll\t\t\t" + format(avg_scores['conll']['avg'], '.2f') + "\n"
    print("AVERAGE METRICS ON DEV SET")
    print(formatted)
    print("AVERAGE LOSS ON DEV SET\n{:.6f}".format(avg_loss))     
    print("Skipped {} documents with zero mentions".format(skipped))
