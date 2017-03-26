""" code to generate vectors for each token in CoNLL document """

import numpy as np
from string import digits, punctuation
import gensim

from conll import ConllCorpusReader

# Google's pre-trained word2vec model from GoogleNews. Note: this 
# is a large matrix loaded into memory and may take several minutes
model = gensim.models.KeyedVectors.load_word2vec_format('/anfs/bigdisc/kh562/Models/GoogleNews-vectors-negative300.bin.gz', binary=True)

# Load CoNLL corpus
corpus_dir = "/anfs/bigdisc/kh562/Corpora/conll-2012/v4/data/"
conll_reader = ConllCorpusReader(corpus_dir)
conll_reader.parse_corpus()
train_conll_docs = conll_reader.get_conll_docs("train")
dev_conll_docs = conll_reader.get_conll_docs("development")

# Translation table to convert digits to #
digit_table = str.maketrans(digits, '#'*len(digits))
# Translation table to remove punctuation
punc_table = str.maketrans('', '', punctuation)

train_matrices=[]
dev_matrices=[]

def get_vector(token):
    """
    Find the index for a token, preprocessing if necessary 
    """
    # Try looking up the token directly
    if token in model:
        return model[token]
    # Otherwise, try replacing digits with #
    token = token.translate(digit_table)
    if token in model:
        return model[token]
    # Otherwise, try removing punctuation
    token = token.translate(punc_table)
    if token in model:
        return model[token]
    # Otherwise, use the null token
    return model['</s>']

# repeat for "test_conll_docs" etc
for conll_doc in dev_conll_docs:
    # list of numpy arrays containing vectors for each token in document
    token_vectors = [get_vector(token) for token in conll_doc.get_document_tokens()]
    # convert list of vectors to matrix
    token_vector_matrix = np.array(token_vectors)
    dev_matrices.append(token_vector_matrix)

np.savez("/anfs/bigdisc/kh562/Corpora/conll-2012/development_docs",matrices=dev_matrices)
