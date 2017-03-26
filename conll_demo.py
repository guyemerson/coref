# HOW TO USE CONLL PARSER

from conll import ConllCorpusReader

corpus_dir = "/anfs/bigdisc/kh562/Corpora/conll-2012/v4/data/"
conll_reader = ConllCorpusReader(corpus_dir)
conll_reader.parse_corpus()
train_conll_docs = conll_reader.get_conll_docs("train")
dev_conll_docs = conll_reader.get_conll_docs("development")

for conll_doc in train_conll_docs:
    # list of all tokens in document
    document_tokens = conll_doc.get_document_tokens()
    # the coref chain consists of a list of coreferent mentions for each key chain id
    coref_chain = conll_doc.coref_chain
    for chain_id in coref_chain:
        mentions = coref_chain[chain_id]
        if len(mentions)==217:
            print document_tokens
        for mention in mentions:
            chain_id = mention.chain_id
            index = mention.get_document_index(conll_doc)
            # this gives the tokens for each mention in document (M in matrix)
            tokens = document_tokens[index[0]:index[1]+1]

