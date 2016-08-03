# HOW TO USE CONLL PARSER

from conll import *

corpus_dir = "/anfs/bigdisc/kh562/Corpora/conll-2011/"
conll_reader = ConllCorpusReader(corpus_dir)
conll_reader.parse_corpus()
train_conll_docs = conll_reader.get_conll_docs("train")
test_conll_docs = conll_reader.get_conll_docs("test")

for conll_doc in train_conll_docs:
    # this contains all the sentences for each document and can be used to create T in matrix
    sents = conll_doc.get_sents()
    # the coref chain consists of a list of coreferent mentions for each key chain id
    coref_chain = conll_doc.get_coref_chain()
    for chain_id in coref_chain:
        mentions = coref_chain[chain_id]
        for mention in mentions:
            chain_id = mention.get_chain_id()
            # this gives the words for each mention in document (M in matrix)
            tokens = mention.get_tokens()
            # the exact index of each mention is below and can be found in "sents"
            # (probably not needed given architecture we're using)
            (sent_index,start_index,end_index) = mention.get_indices()
