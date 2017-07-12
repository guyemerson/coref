import numpy as np
import subprocess

METRICS = ['muc', 'bcub', 'ceafm', 'ceafe', 'blanc', 'lea']

def get_evaluation(gold_doc, coref_mat, threshold):
    chain_ids = sorted(gold_doc.coref_chain.keys())
    gold_mentions = [m for chain_id in chain_ids for m in gold_doc.coref_chain[chain_id]]
    # binarise matrix according to threshold
    binarised = np.array(np.where(coref_mat > threshold, 1, 0))
    # make sure mentions are self-referential
    np.fill_diagonal(binarised, 1)
    # check if any similarity greater than threshold
    if np.array_equal(binarised, np.identity(binarised.shape[1])):
        return null_scores()
    # remove singleton mentions
    predictions = np.vstack({tuple(row) for row in binarised if np.sum(row)>1})
    # join overlapping predicted coref chains
    for col in range(0, predictions.shape[1]):
        overlaps = None if len(np.where(predictions.T[col]==1)[0]) == 1 else list(np.where(predictions.T[col]==1)[0])
        if overlaps:
            predictions = np.vstack((predictions, np.logical_or.reduce([predictions[i] for i in overlaps])))
            predictions = np.delete(predictions, overlaps, axis=0)
    predicted_coref_chain = {}
    for chain_id, chain in enumerate(predictions):
        mentions = [gold_mentions[i] for i,v in enumerate(chain) if v == 1]
        predicted_coref_chain[str(chain_id)] = mentions
    gold = populate_doc(gold_doc, gold_doc.coref_chain)    
    test = populate_doc(gold_doc, predicted_coref_chain)
    write_doc("gold", gold)
    write_doc("test", test)
    return get_scores("gold", "test")

def populate_doc(doc, coref_chain):
    doc_array = []
    for sent_index, sent in enumerate(doc.sents):
        doc_array.append([])
        for word in sent:
            doc_array[sent_index].append(word)
    for chain_id in coref_chain:
        for mention in coref_chain[chain_id]:
            sent_id, start_id, end_id = mention.get_indices()
            prefix_s = "|" if len(doc_array[sent_id][start_id].split("\t")) > 1 else "\t"
            prefix_e = "|" if len(doc_array[sent_id][end_id].split("\t")) > 1 else "\t"
            if start_id == end_id: 
                doc_array[sent_id][start_id] += prefix_s+"("+chain_id+")"
            else: 
                doc_array[sent_id][start_id] += prefix_s+"("+chain_id
                doc_array[sent_id][end_id] += prefix_e+chain_id+")"
    return doc_array

def write_doc(name, doc):
    output = open(name,"w")
    output.write("#begin document (test);\n")
    for sent in doc:
        for i, word_info in enumerate(sent):
            word, coref = (word_info,"-") if len(word_info.split("\t")) == 1 else word_info.split("\t")
            output.write("test\t0\t%d\t%s\t%s\n" % (i, word, coref))
        output.write("\n")
    output.write("#end document")

def format_scores(scores):
    lines = ["\tR\tP\tF1"]
    for metric in METRICS:
        lines.append(metric + "\t"
                     + "\t".join([str(val) for val in scores[metric]]))
    lines.append("avg conll\t\t\t"
                 + format(scores["avg"], '.2f'))
    return "\n".join(lines)

def null_scores():
    scores = {metric : (0., 0., 0.) for metric in METRICS}
    scores["avg"] = 0.
    scores["formatted"] = format_scores(scores)
    return scores

def get_scores(gold, test):
    scorer_output = subprocess.check_output(["perl",
                                             "/anfs/bigdisc/kh562/LEA-coreference-scorer/scorer.pl",
                                             "all",
                                             gold,
                                             test,
                                             "none"]).decode()
    scores = {}
    metric = None
    for line in scorer_output.split("\n"):
        if not line: continue
        tokens = line.split()
        if tokens[0] == "METRIC":
            metric = line.split()[1][:-1]
        if (metric != 'blanc' and line.startswith("Coreference:")) \
           or (metric == 'blanc' and line.startswith("BLANC:")):
            scores[metric] = (float(tokens[5][:-1]), float(tokens[10][:-1]), float(tokens[12][:-1]))
    scores["avg"] = (scores["muc"][2]
                     + scores["bcub"][2]
                     + scores["ceafe"][2]
                     )/3
    scores["formatted"] = format_scores(scores)
    return scores

def harmonic_mean(x, y):
    return 2*x*y/(x+y)
