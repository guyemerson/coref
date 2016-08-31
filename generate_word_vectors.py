""" code to generate vectors for each token in CoNLL document """

import numpy as np
from gensim.models import Word2Vec
from conll import *

# google's pre-trained word2vec model from GoogleNews. Note: this 
# is a large matrix loaded into memory and may take several minutes
model = Word2Vec.load_word2vec_format('/anfs/bigdisc/kh562/Models/GoogleNews-vectors-negative300.bin.gz', binary=True)

corpus_dir = "/anfs/bigdisc/kh562/Corpora/conll-2011/"
conll_reader = ConllCorpusReader(corpus_dir)
conll_reader.parse_corpus()
train_conll_docs = conll_reader.get_conll_docs("train")
test_conll_docs = conll_reader.get_conll_docs("test")

# repeat for "test_conll_docs" etc
for conll_doc in train_conll_docs:
    # list of numpy arrays containing vectors for each token in document
    token_vectors=[]
    # list of all tokens in document
    for token in conll_doc.get_document_tokens():
        # when token is not in model, I create empty vector of length 300 (dim of Google's model)
        token_vectors.append(np.array([float(0)]*300,dtype=np.float32) if token not in model else model[token])
