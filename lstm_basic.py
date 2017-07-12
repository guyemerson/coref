#!/usr/bin/env python3

# Starting to implement Cheng & Voigt

import numpy as np
import tensorflow as tf
import argparse
import os.path

from conll import ConllCorpusReader
from matrix_gen import get_s_matrix, coref_matrix
from evaluation import get_evaluation, METRICS, format_scores, harmonic_mean

parser = argparse.ArgumentParser()
parser.add_argument("--print_document_scores", help="Print metrics per document during training", action="store_true")
parser.add_argument("--print_minibatch_loss", help="Print minibatch (document) loss during training", action="store_true")
parser.add_argument("--print_coref_matrices", help="Print gold and predicted coreference matrices", action="store_true")
parser.add_argument("--print_dev_loss", help="Print the metrics/loss/matrices during evaluation on the dev set", action="store_true")
parser.add_argument("--epochs", help="Number of training epochs", default=20)
parser.add_argument("--learning_rate", help="Learning rate for training", default=0.001)
parser.add_argument("--hidden_size", help="Number of hidden units", default=100)
parser.add_argument("--threshold", help="Threshold value for coference (between 0 and 1)", default=0.79, type=float)
parser.add_argument("--reg_weight", help="The weight of regularization function", default=10**-7, type=float)
parser.add_argument("--additional_features",help="Use vectors containing additional features",action="store_true")
parser.add_argument("--model_dir", help="Directory for saving models", default="models")
parser.add_argument("--saved_model", help="Restore a trained model. To evaluate without further training, set epochs to 0")
args = parser.parse_args()

corpus_dir = "/anfs/bigdisc/kh562/Corpora/conll-2012/"
conll_reader = ConllCorpusReader(corpus_dir).parse_corpus()

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

cost =  tf.truediv(-tf.reduce_sum(y*tf.log(nonneg_sim+(1e-5))
                                  + (1-y)*tf.log(1+(1e-5)-nonneg_sim)),
                   tf.cast(tf.size(y), tf.float32), name="op_cost")
reg = tf.multiply(REGULARIZATION_WEIGHT, sum([tf.reduce_sum(x**2)
                                              for x in tf.trainable_variables()]),
                  name="op_reg")
regcost = tf.add(cost, reg, name="op_regcost")

# TODO: currently the importance of a document grows quadratically with the number of referring expressions

# Train the model
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(regcost)

init = tf.global_variables_initializer()

### Load data

# cached document vectors
# latin1 encoding because they were generated with Python 2
if args.additional_features:
    train_embeddings = np.load(os.path.join(corpus_dir, "training_docs_new.npz"), encoding='latin1')["matrices"]
    dev_embeddings = np.load(os.path.join(corpus_dir, "development_docs_new.npz"), encoding='latin1')["matrices"]
    test_embeddings = np.load(os.path.join(corpus_dir, "test_docs_new.npz"), encoding='latin1')["matrices"]
else:
    train_embeddings = np.load(os.path.join(corpus_dir, "training_docs.npz"), encoding='latin1')["matrices"]
    dev_embeddings = np.load(os.path.join(corpus_dir, "development_docs.npz"), encoding='latin1')["matrices"]
    test_embeddings = np.load(os.path.join(corpus_dir, "test_docs.npz"), encoding='latin1')["matrices"]

train_conll_docs = conll_reader.get_conll_docs("train")
train_s_matrices = [get_s_matrix(x) for x in train_conll_docs]
train_coref_matrices = [coref_matrix(x) for x in train_conll_docs]

dev_conll_docs = conll_reader.get_conll_docs("development")
dev_s_matrices = [get_s_matrix(x) for x in dev_conll_docs]
dev_coref_matrices = [coref_matrix(x) for x in dev_conll_docs]

# To save the trained model
saver = tf.train.Saver()

def process_data(sess, dataset, epoch=None):
    """
    Evaluate on data, and update weights if using training set
    """
    # Choose dataset
    if dataset == 'train':
        documents = train_conll_docs
        embeddings = train_embeddings
        s_matrices = train_s_matrices
        coref_matrices = train_coref_matrices
        optim = True
    elif dataset == 'dev':
        documents = dev_conll_docs
        embeddings = dev_embeddings
        s_matrices = dev_s_matrices
        coref_matrices = dev_coref_matrices
        optim = False
    else:
        raise ValueError(dataset)
    
    metrics = {m : ([], [], [])  # lists of recall, precision, F1
               for m in METRICS}
    metrics['avg'] = []
    losses = []
    skipped = 0  # number of empty documents
    output_nodes = [nonneg_sim, cost]
    if optim:  # only use the optimizer for the training set
        output_nodes.append(optimizer)
    
    # Process one document at a time
    for i, (doc, embed_mat, s_mat, coref_mat) in enumerate(zip(documents, embeddings, s_matrices, coref_matrices)):
        # Skip empty documents
        if coref_mat.size == 0:
            skipped += 1
            continue
        # Train
        current_dict = {x: embed_mat,
                        y: coref_mat,
                        s: s_mat}
        predicted_coref, loss, *_ = sess.run(output_nodes, feed_dict=current_dict)
        # Record loss
        losses.append(loss)
        # Use standard coreference metrics
        try:  # avoid errors at some thresholds
            evals = get_evaluation(doc, predicted_coref, THRESHOLD)
        except:
            continue
        for m in METRICS:
            for j in range(3):  # recall, precision, f1
                metrics[m][j].append(evals[m][j])
        metrics['avg'].append(evals['avg'])
        # Print
        if dataset == 'train' or args.print_dev_loss:
            if epoch is not None:
                print('Epoch {}'.format(epoch+1))
            print('Document {}'.format(i+1))
            if args.print_coref_matrices:
                print("GOLD COREFERENCE MATRIX")
                print(coref_mat)
                print("PREDICTED COREFERENCE MATRIX")
                print(predicted_coref)
            if args.print_document_scores:
                print(evals["formatted"])
            if args.print_minibatch_loss:
                print("Minibatch loss {:.6f}".format(loss))
    
    # Average training error across documents
    avg_loss = sum(losses) / len(losses)
    avg_scores = {m : [sum(metrics[m][j]) / len(metrics[m][j])
                       for j in range(3)]
                  for m in METRICS}
    avg_scores['avg'] = sum(metrics['avg']) / len(metrics['avg'])
    # Note that the f1 of the mean is not the mean of the f1
    avg_scores['conll'] = (harmonic_mean(*avg_scores['muc'][:2])
                           + harmonic_mean(*avg_scores['bcub'][:2])
                           + harmonic_mean(*avg_scores['ceafe'][:2])
                           )/3
    if epoch is not None:
        print("END OF EPOCH {}".format(epoch+1))
    print("AVERAGE METRICS")
    print(format_scores(avg_scores))
    print('conll\t\t\t' + format(avg_scores['conll'], '.2f'))
    print("AVERAGE LOSS {:.6f}".format(avg_loss))
    print("Skipped {} documents with zero mentions".format(skipped))
    
    return avg_loss, avg_scores


### Start session

with tf.Session() as sess:
    print("Starting session")
    sess.run(init)
    if args.saved_model:
        saver.restore(args.saved_model)
    
    for step in range(EPOCHS):
        # Train the model
        process_data(sess, 'train', step)
        # Save
        print("Saving model")
        saver.save(sess,
                   os.path.join(args.model_dir,
                                '_'.join(['model',
                                          str(LEARNING_RATE),
                                          str(EPOCHS),
                                          str(NUM_HIDDEN),
                                          str(REGULARIZATION_WEIGHT),
                                          'epoch' + str(step+1)])))
        # Evaluate on devset
        process_data(sess, 'dev')
    
    if EPOCHS == 0 and args.saved_model:
        process_data(sess, 'dev')
