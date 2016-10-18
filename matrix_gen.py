# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 15:20:56 2016

@author: Olesya Razuvayevskaya, Guy Emerson
"""

import numpy as np

def get_s_matrix(doc):
    # Find number of tokens and initialize list of mention vectors
    n_toks = len(doc.get_document_tokens())
    S_list = []
    # The coref chain consists of a list of coreferent mentions for each key chain id
    # Sort the ids so that the order is stable
    chains = sorted(doc.coref_chain.keys())
    # Generate the S_matrix
    for chain_id in chains:
        for mention in doc.coref_chain[chain_id]:
            #initialize the matrix row
            row = np.zeros(n_toks)
            start_index, end_index = mention.get_document_index(doc)
            N = end_index+1 - start_index
            # assign values to the mention tokens in the tokens vector
            row[start_index:end_index+1] = 1/N
            S_list.append(row)
    return np.array(S_list)

def total_chain_len(doc,item):
    length=0
    for chain in item:
        length=length+len(doc.coref_chain[chain])
    return(length)
 
    
def coref_matrix(doc):
    C_list=[]
    chains = sorted(doc.coref_chain.keys())
    n_mentions = total_chain_len(doc,chains)
    start_index=0
    for chain_id in chains:
        end_index=start_index+len(doc.coref_chain[chain_id])
        for mention in doc.coref_chain[chain_id]:  
            row = np.zeros(n_mentions)
            row[start_index:end_index]=1
            C_list.append(row)
        start_index=end_index
    return(np.array(C_list))


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
