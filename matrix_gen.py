# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 15:20:56 2016

@author: Olesya Razuvayevskaya, Guy Emerson
"""

import numpy as np

def get_s_matrix(doc):
    """
    Return a matrix of shape [num_mentions, num_tokens],
    with a nonzero value whenever the token is part of the mention.
    Each row sums to 1. 
    :param doc: ConllDocument
    :return: numpy array
    """
    # Find number of tokens and initialize list of mention vectors
    n_toks = len(doc.get_document_tokens())
    S_list = []
    # The coref chain consists of a list of coreferent mentions for each key chain id
    # Sort the ids so that the order is stable
    chains = sorted(doc.coref_chain.keys())
    # Generate the S_matrix
    for chain_id in chains:
        for mention in doc.coref_chain[chain_id]:
            # Initialize the matrix row
            row = np.zeros(n_toks)
            start_index, end_index = mention.get_document_index(doc)
            N = end_index+1 - start_index
            # assign values to the mention tokens in the tokens vector
            row[start_index:end_index+1] = 1/N
            S_list.append(row)
    if S_list:
        return np.array(S_list)
    return np.zeros((0,n_toks))

def get_mention_matrix(doc):
    """
    Return a matrix of shape [num_mentions, num_tokens],
    with a value of 1 at the final token of each mention. 
    :param doc: ConllDocument
    :return: numpy array
    """
    # Find number of tokens and initialize list of mention vectors
    n_toks = len(doc.get_document_tokens())
    S_list = []
    # The coref chain consists of a list of coreferent mentions for each key chain id
    # Sort the ids so that the order is stable
    chains = sorted(doc.coref_chain.keys())
    # Generate the S_matrix
    for chain_id in chains:
        for mention in doc.coref_chain[chain_id]:
            # Initialize the matrix row as 0s
            row = np.zeros(n_toks, dtype='bool')
            # Set 1 at the final token
            _, end_index = mention.get_document_index(doc)
            row[end_index] = 1
            S_list.append(row)
    if S_list:
        return np.array(S_list)
    return np.zeros((0,n_toks))
    
def coref_matrix(doc):
    """
    Return a matrix of shape [num_mentions, num_mentions],
    with a value of 1 whenever two mentions are coreferent. 
    :param doc: ConllDocument
    :return: numpy array
    """
    C_list=[]
    chains = sorted(doc.coref_chain.keys())
    n_mentions = sum([len(doc.coref_chain[chain]) for chain in chains])
    start_index=0
    for chain_id in chains:
        end_index=start_index+len(doc.coref_chain[chain_id])
        for mention in doc.coref_chain[chain_id]:  
            row = np.zeros(n_mentions)
            row[start_index:end_index]=1
            C_list.append(row)
        start_index=end_index
    if C_list:
        return(np.array(C_list))
    return np.zeros((0,0))


if __name__ == "__main__":
    # Check it works
    from conll import ConllCorpusReader
    corpus_dir = "/anfs/bigdisc/kh562/Corpora/conll-2011/"
    conll_reader = ConllCorpusReader(corpus_dir).parse_corpus()
    train_conll_docs = conll_reader.get_conll_docs("train")
    s_matrix = get_s_matrix(train_conll_docs[0])
    nonzero, = s_matrix[0].nonzero()
    print(nonzero)
    print(train_conll_docs[0].get_document_tokens()[nonzero.min():nonzero.max()+1])
