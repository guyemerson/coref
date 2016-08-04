# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 15:20:56 2016

@author: Olesya Razuvayevskaya
"""
#from __future__ import division

from conll import *
import string


class matrix_generator:
    def get_s_matrix(self, doc):
        T_matrix=[]
        S_matrix=[]
        sents = doc.get_sents()
        # generate the T matrix
        for sent in sents:
            words = [w for w in sent if not w in set(string.punctuation)] #have to discuss whether we need to lowercase and preprocess words
            for token in words:
                if not token in T_matrix:
                    T_matrix.append(token)
        # the coref chain consists of a list of coreferent mentions for each key chain id
        coref_chain = doc.get_coref_chain()
        #generate the S_matrix
        for chain_id in coref_chain:
            mentions = coref_chain[chain_id]
            for mention in mentions:
                row=list(T_matrix)
                tokens = [w for w in mention.get_tokens() if not w in set(string.punctuation)]
                for i, token in enumerate(row):
                    if not token in tokens:
                        row[i]=0
                    else:
                        row[i]=1/len(tokens)
                S_matrix.append(row)
        return(S_matrix)