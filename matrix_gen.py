# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 15:20:56 2016

@author: Olesya Razuvayevskaya
"""
#from __future__ import division

from conll import *
import string


class matrix_generator:
    def get_s_matrix(doc):
    # generate the T matrix and initialize S_matrix
    T_matrix=doc.get_document_tokens()
    S_matrix=[]
    # the coref chain consists of a list of coreferent mentions for each key chain id
    coref_chain = doc.coref_chain
    #generate the S_matrix
    for chain_id in coref_chain:
        mentions = coref_chain[chain_id]
        for mention in mentions:
            #initialize the matrix row
            row=[0]*len(T_matrix)
            chain_id = mention.chain_id
            index = mention.get_document_index(doc)
            # assign values to the mention tokens in the tokens vector
            for i in range(index[0],(index[1]+1)):
                row[i]=1/(index[1]+1-index[0])
            S_matrix.append(row)
    return(S_matrix)
