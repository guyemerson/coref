import numpy as np

def get_s_matrix(doc):
    """
    Return a matrix of shape [num_mentions, num_tokens],
    with a nonzero value whenever the token is part of the mention.
    Each row sums to 1. 
    :param doc: ConllDocument
    :return: numpy array
    """
    # Find number of tokens and initialize matrix
    n_toks = doc.get_n_tokens()
    n_mentions = doc.get_n_mentions()
    matrix = np.zeros((n_mentions, n_toks))
    # Generate the S_matrix
    # The coref chain consists of a list of coreferent mentions for each key chain id
    # Use sorted chains so that the order is stable
    for i, mention in enumerate(doc.iter_mentions()):
        start_index, end_index = mention.get_document_index(doc)
        end_index += 1  # need the index *after* the last token
        N = end_index - start_index
        # assign positive values for the mention's tokens
        matrix[i, start_index:end_index] = 1/N
    return matrix

def get_mention_matrix(doc, n_mentions=None):
    """
    Return a matrix of shape [num_tokens, num_mentions],
    with a value of 1 at the final token of each mention. 
    :param doc: ConllDocument
    :param n_mentions: if specified, pad up to a given number of mentions
    :return: numpy array
    """
    # Find number of tokens and initialize matrix
    n_toks = doc.get_n_tokens()
    doc_mentions = doc.get_n_mentions()
    if n_mentions is None:
        n_mentions = doc_mentions
    else:
        n_mentions = max(n_mentions, doc_mentions)
    matrix = np.zeros((n_toks, n_mentions), dtype='int')
    # Generate the matrix
    for i, mention in enumerate(doc.mentions):
        start_index, end_index = mention.get_document_index(doc)
        # Set 1 for each token
        matrix[start_index:end_index+1, i] = 1
        # Set 2 at the beginning token
        matrix[start_index, i] = 2
        # Set 3 at the end token
        matrix[end_index, i] = 3
    return matrix

def get_beginning_inside_end(doc, *args, **kwargs):
    """
    Return matrices of shape [num_tokens, num_mentions], with a value of 1 at:
    - the first token of each mention
    - all tokens of each mention
    - the last token of each mention
    :param doc: ConllDocument
    :return: numpy arrays
    """
    mat = get_mention_matrix(doc, *args, **kwargs)
    return mat == 2, mat > 0, mat == 3

def get_attachment_matrix(doc):
    """
    Return a matrix of shape [num_mentions, num_mentions],
    with a 1 in each row, for the first mention in the document that it's coreferent with. 
    :param doc: ConllDocument
    :return: numpy array
    """
    n_mentions = doc.get_n_mentions()
    matrix = np.zeros((n_mentions, n_mentions))
    
    # Map from chain's ID to the first mention's ID
    chain_to_first = {chain_id: mentions[0].mention_id
                      for chain_id, mentions in doc.coref_chain.items()}
    
    for mention in doc.mentions:
        matrix[mention.mention_id, chain_to_first[mention.chain_id]] = 1
    
    return matrix
    
def coref_matrix(doc):
    """
    Return a matrix of shape [num_mentions, num_mentions],
    with a value of 1 whenever two mentions are coreferent. 
    :param doc: ConllDocument
    :return: numpy array
    """
    n_mentions = doc.get_n_mentions()
    matrix = np.zeros((n_mentions, n_mentions))
    i = 0
    for chain_id in doc.get_sorted_chains_ids():
        chain_length = len(doc.coref_chain[chain_id])
        if chain_length:  # in case of empty chains
            j = i+chain_length
            matrix[i:j, i:j] = 1
            i = j
    return matrix


if __name__ == "__main__":
    # Check it works
    from conll import ConllCorpusReader
    corpus_dir = "/anfs/bigdisc/kh562/Corpora/conll-2012/"
    conll_reader = ConllCorpusReader(corpus_dir).parse_corpus()
    train_conll_docs = conll_reader.get_conll_docs("train")
    s_matrix = get_s_matrix(train_conll_docs[0])
    attachment_matrix = get_attachment_matrix(train_conll_docs[0])
    nonzero, = s_matrix[0].nonzero()
    print(nonzero)
    print(train_conll_docs[0].get_document_tokens()[nonzero.min():nonzero.max()+1])
