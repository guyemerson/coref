""" code to generate vectors for each token in CoNLL document """

import numpy as np
import gensim
from string import digits, punctuation
from conll import ConllCorpusReader

# Google's pre-trained word2vec model from GoogleNews. Note: this 
# is a large matrix loaded into memory and may take several minutes
model = gensim.models.KeyedVectors.load_word2vec_format('/anfs/bigdisc/kh562/Models/GoogleNews-vectors-negative300.bin.gz', binary=True)

# Load CoNLL corpus
corpus_dir = "/anfs/bigdisc/kh562/Corpora/conll-2012/"
conll_reader = ConllCorpusReader(corpus_dir)
conll_reader.parse_corpus()
train_conll_docs = conll_reader.get_conll_docs("train")
dev_conll_docs = conll_reader.get_conll_docs("development")
test_conll_docs = conll_reader.get_conll_docs("test")

# Translation table to convert digits to #
digit_table = str.maketrans(digits, '#'*len(digits))
# Translation table to remove punctuation
punc_table = str.maketrans('', '', punctuation)
# load upenn pos tagset
upenn_tagset = [l.strip() for l in open("upenn_tagset.txt","r").readlines()]

# maximum number of speakers in CoNLL corpus
MAX_SPEAKERS = 10

train_matrices, dev_matrices, test_matrices = [],[],[]

def get_token_vectors(conll_doc):
    """
    Find the vector for a token, preprocessing if necessary 
    """
    pos_tags = conll_doc.pos_tags
    speakers = conll_doc.speakers
    speaker_list = sorted(list(set(speakers)))
    # remove cases where no speaker
    if "-" in speaker_list:
        speaker_list.remove("-")
    token_index = 0
    token_vectors = []
    for sent in conll_doc.sents:
        for i,token in enumerate(sent):
            # Try looking up the token directly
            if token in model:
                vector = model[token]
            else:
                # Otherwise, try replacing digits with #
                token = token.translate(digit_table)
                if token in model:
                    vector = model[token]
                else:
                    # Otherwise, try removing punctuation
                    token = token.translate(punc_table)
                    if token in model:
                        vector =  model[token]
                    # Otherwise, use the null token
                    else: vector = model['</s>']
            # check if this is the last token in sent
            vector = np.append(vector,1 if i==len(sent)-1 else 0)
            # check if token is capitalised
            vector = np.append(vector,1 if token.isupper() else 0)
            # get binary pos tag feature
            vector = np.append(vector,[0 if i!=upenn_tagset.index(pos_tags[token_index]) else 1 for i in range(len(upenn_tagset))])
            # get binary presence of speaker n. "-" indicates no speaker
            if speakers[token_index]=="-":
                vector = np.append(vector,[0] * MAX_SPEAKERS)
            else:
                vector = np.append(vector,[0 if i!=speaker_list.index(speakers[token_index]) else 1 for i in range(MAX_SPEAKERS)])
            token_vectors.append(vector)
            token_index+=1
    return token_vectors

# repeat for "test_conll_docs" etc
for conll_doc in train_conll_docs:
    # list of numpy arrays containing vectors for each token in document
    token_vectors = get_token_vectors(conll_doc)
    # convert list of vectors to matrix
    token_vector_matrix = np.array(token_vectors)
    train_matrices.append(token_vector_matrix)

np.savez("/anfs/bigdisc/kh562/Corpora/conll-2012/training_docs_new",matrices=train_matrices)
