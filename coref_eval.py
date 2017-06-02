#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import argparse
from collections import defaultdict

from conll import ConllCorpusReader
from matrix_gen import get_s_matrix, coref_matrix
from evaluation import get_evaluation

parser = argparse.ArgumentParser()
parser.add_argument("--print_coref_matrices", help="Print gold and predicted coreference matrices", action="store_true")
parser.add_argument("--model", help="Base name of saved model", default="/local/filespace/lr346/disco/experiments/coref/models/model_0.001_20_100_0.0_epoch19")
parser.add_argument("--additional_features",help="Use vectors containing additional features",action="store_true")
parser.add_argument("--threshold", help="Minimum cutoff for positive coreference", default=0.6)
args = parser.parse_args()

corpus_dir = "/anfs/bigdisc/kh562/Corpora/conll-2012/"
conll_reader = ConllCorpusReader(corpus_dir).parse_corpus()

metrics = ['muc', 'bcub', 'ceafm', 'ceafe', 'blanc', 'lea']

INPUT_SIZE = 362 if args.additional_features else 300

print("Starting session")
with tf.Session() as sess:

    print("Retrieving saved model")
    saver = tf.train.import_meta_graph(args.model + '.meta')
    saver.restore(sess, args.model)

    print("Retrieving placeholders and prediction operation")
    x = tf.get_collection('x')[0]
    y = tf.get_collection('y')[0]
    s = tf.get_collection('s')[0]
    nonneg_sim = tf.get_default_graph().get_tensor_by_name("op_dot_product:0")

    print("Loading dev data")
    dev_docs = np.load("/anfs/bigdisc/kh562/Corpora/conll-2012/development_docs.npz", encoding='latin1')["matrices"]
    dev_conll_docs = conll_reader.get_conll_docs("development")
    dev_s_matrix = [get_s_matrix(x) for x in dev_conll_docs]
    dev_coref_matrix = [coref_matrix(x) for x in dev_conll_docs]

    print("Evaluating on dev set")
    print("\n")
    skipped = 0
    metrics_on_dev = defaultdict(lambda: defaultdict(list))
    for i in range(len(dev_conll_docs)):
        current_dict = {x: dev_docs[i], y: dev_coref_matrix[i], s: dev_s_matrix[i]}
        if dev_coref_matrix[i].size == 0:
            skipped += 1
            continue
        coref_mat = sess.run(nonneg_sim, feed_dict=current_dict)
        evals = get_evaluation(dev_conll_docs[i],coref_mat,args.threshold)
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
    formatted = "\tR\tP\tF1\n"
    for m in metrics:
        formatted += m + "\t" + format(avg_scores[m]['R'], '.2f') + "\t" +  format(avg_scores[m]['P'], '.2f') + "\t" + format(avg_scores[m]['F1'], '.2f') + "\n"
    formatted += "\n"
    formatted += "conll\t\t\t" + format(avg_scores['conll']['avg'], '.2f') + "\n"
    print("AVERAGE METRICS ON DEV SET")
    print(formatted)
    print("Skipped {} documents with zero mentions".format(skipped))
