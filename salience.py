import numpy as np
import tensorflow as tf

class SalienceMemoryMixin(tf.contrib.rnn.RNNCell):
    """
    A salience memory extension to be added to any standard RNN cell.
    Note that this class is not directly usable;
    rather, it makes several methods available for use by a derivative RNN cell class
    """
    
    def __init__(self, max_mentions, *args, memory_size=None, salience_decay=0.999, salience_drop=0.001, new_entity_score=0.5, attention_beta=5., **kwargs):
        """
        Set sizes of tensors and initialise weights
        :param max_mentions: maximum number of mentions in an input sequence
        Remaining arguments and keyword arguments are passed to super
        """
        # Initialise the standard RNN
        super().__init__(*args, **kwargs)
        # Sizes of objects
        self.max_mentions = max_mentions
        # If memory size is not specified, use the same size as the parent class
        if memory_size is None:
            self.memory_size = super().output_size
        else:
            self.memory_size = memory_size
        # Special parameters
        # TODO - put these into scope? Make them variables?
        self.salience_decay = tf.constant(salience_decay, tf.float64)
        self.salience_drop = tf.constant(salience_drop, tf.float64)
        self.new_entity_score = tf.constant(new_entity_score, tf.float64)
        self.attention_beta = tf.constant(attention_beta, tf.float64)
    
    @property
    def state_size(self):
        """
        Return the shapes of the hidden state, including the memory extension
        """
        return ((tf.TensorShape([self.max_mentions, self.memory_size]),  # Memory vectors
                 tf.TensorShape([self.max_mentions]),  # Salience of memories
                 tf.TensorShape([])),  # Index of next free memory  
                super().state_size)  # State size of parent RNN cell
    
    def read_memory(self, state, query):
        """
        Read from the memory extension
        :param state: tuple of state tensors (memory, salience, index), with batch size of 1
        :param query: batch of vectors to match against the memory
        :return: attention over memories, retrieved memory weighted by attention
        """
        # Get number of query vectors
        n_queries = tf.shape(query)[0]
        # Unpack state, ignoring the parent RNN
        memory, salience, index_float = state
        # Cast index from float to int
        index = tf.cast(index_float, tf.int32)
        # Cosine similarity between query and memories
        normed_query = tf.nn.l2_normalize(query, dim=1)
        normed_memory = tf.nn.l2_normalize(memory, dim=2)
        # (squeeze to deal with batch of size 1)
        similarity = tf.matmul(normed_query,  # [batch_size, memory_size]
                               tf.squeeze(normed_memory, 0),  # [1, max_mentions, memory_size]
                               transpose_b=True)  # gives [batch_size, max_mentions]
        weighted_sim = tf.nn.relu(similarity) * salience  # negative similarity might interact strangely with salience
        # Take attention over these similarities,
        # with a default score for the new entity
        # and no attention for later entities
        # (An alternative attention would involve two decisions:)
        # (whether to use an existing entity, and if so, which  )
        # TODO - what if a multiple mentions finish on a token, and one of those starts a new chain?
        infs = -np.inf * tf.ones((n_queries, self.max_mentions), dtype=tf.float64)
        single_mask = tf.sequence_mask([index[0]+1], self.max_mentions)
        mask = tf.tile(single_mask, [n_queries, 1])
        scores = tf.where(mask, weighted_sim, infs) + tf.one_hot(index, self.max_mentions, self.new_entity_score, dtype=tf.float64)
        attention = tf.nn.softmax(scores * self.attention_beta)  # [1, max_mentions]
        # Retrieve a memory as a weighted sum, and update the hidden state
        retrieved_memory = tf.matmul(attention, tf.squeeze(memory, 0))
        
        return attention, retrieved_memory
    
    def write_memory(self, state, attention, update, n_updates):
        """
        Write to the memory extension
        :param state: tuple of state tensors (memory, salience, index), with batch size of 1:
            [1, max_mentions, memory_size], [1, max_mentions], [1]  
        :param attention: batch of distributions over memories
            [batch_size, max_mentions] 
        :param update: batch of vectors to add to memories
            [batch_size, memory_size]
        :param n_updates: number of actual updates in the batch (TODO - tidy this)
        :return: new state
        """
        # Unpack state, ignoring the parent RNN
        memory, salience, index = state
        # Add the update to the memory
        expanded_update = tf.expand_dims(attention, 2) * tf.expand_dims(update, 1)  # Broadcasting for outer product
        new_memory = memory + tf.reduce_sum(expanded_update, 0)  # Sum over batch
        # Update the salience
        new_salience = tf.maximum(salience, tf.reduce_max(attention, 0))
        # Increment the index
        new_index = index + n_updates
        
        return new_memory, new_salience, new_index
    
    def decay_salience(self, state):
        """
        Decay the salience, allowing memories to be slowly forgotten
        :param state: tuple of state tensors
        :return: new state
        """
        memory, salience, index = state
        new_salience = tf.maximum(salience * self.salience_decay - self.salience_drop, 0)
        return memory, new_salience, index


class EntityMixin(SalienceMemoryMixin):
    """
    A class for extending an RNN cell with:
    - a salience memory bank, to represent discourse entities
    - a working memory, to process each mention of some discourse entity
    """
    
    def __init__(self, *args, SecondaryCellClass=tf.contrib.rnn.LSTMCell, secondary_cell_kwargs={}, **kwargs):
        # Initialise main RNN
        super().__init__(*args, **kwargs)
        # Initialise working memory RNN
        # Note that for the secondary cell, the batch size is the maximum number of mentions
        with tf.variable_scope('secondary_cell'):
            self.secondary_cell = SecondaryCellClass(self.memory_size, **secondary_cell_kwargs)
        # Additional weights
        self.hidden_to_query = tf.Variable(tf.random_normal((self.memory_size, self.memory_size), stddev=0.01, dtype=tf.float64))
        self.working_to_query = tf.Variable(tf.random_normal((self.memory_size, self.memory_size), stddev=0.01, dtype=tf.float64))
        self.hidden_to_init_working = tf.Variable(tf.random_normal((self.memory_size, self.memory_size), stddev=0.01, dtype=tf.float64))
    
    @property
    def state_size(self):
        """
        Return the shapes of the hidden state, including the memory extension
        """
        expanded_shapes = []
        for shape in self.secondary_cell.state_size:
            if isinstance(shape, tf.TensorShape):
                expanded_shapes.append(tf.TensorShape([self.max_mentions, *shape]))
            else:
                expanded_shapes.append(tf.TensorShape([self.max_mentions, shape]))
        return (expanded_shapes,  # Working memory vectors
                *super().state_size)  # State size of parent RNN cell
    
    @property
    def output_size(self):
        """
        Return the shapes of the output, including the memory extension
        """
        return (tf.TensorShape([self.max_mentions, self.max_mentions]),  # Working memory retrieval decisions
                super().output_size)  # Output size of parent RNN cell
        
    
    def __call__(self, inputs, state, scope=None):
        """
        Run the RNN
        :param inputs: (embeddings, beginning bools, inside bools, end bools), all with batch size 1
        :param state: tuple of state tensors
        :param scope: VariableScope for the created subgraph
        :return: output, new_state
        """
        # Unpack input and state
        embedding, beginning, inside, end = inputs
        working_state, memory_state, inner_state = state
        hidden = inner_state[-1]  # Assume output is last element of tuple
        
        # Query the salience memory (residual connection), then call the parent RNN
        query = tf.tanh(hidden + tf.matmul(hidden, self.hidden_to_query))
        _, response = self.read_memory(memory_state, query)
        embedding_and_response = tf.concat([embedding, response], 1)
        inner_output, new_inner_state = super().__call__(embedding_and_response, inner_state, scope)
        
        # Working memory
        
        # Remove batch size of 1
        sq_working = [tf.squeeze(x, 0) for x in working_state]
        prev_output = sq_working[-1]  # Assume output is last element of tuple
        # Initialise new mentions (residual connection)
        init_vec = inner_output + tf.matmul(inner_output, self.hidden_to_init_working)
        prev_output += tf.transpose(beginning) * init_vec  # Broadcasting for outer product
        # Query the salience memory (residual connection), then call the secondary RNN
        queries = tf.tanh(prev_output + tf.matmul(prev_output, self.working_to_query))
        _, responses = self.read_memory(memory_state, queries)
        hidden_and_response = tf.concat([tf.tile(inner_output, [self.max_mentions, 1]),
                                         responses], 1)
        with tf.variable_scope('secondary_cell'):
            working_output, new_sq_working = self.secondary_cell(hidden_and_response, sq_working)
        # Add batch size of 1, and keep state only when inside the mention
        new_working_state = [tf.expand_dims(x * tf.transpose(inside), 0)
                             for x in new_sq_working]
        # Make attachment decisions at the end of each mention
        # TODO use tf.boolean_mask?
        #shrunk_working_output = tf.boolean_mask(working_output,
        #                                        tf.cast(tf.squeeze(end, 0), tf.bool))
        attach_queries = tf.tanh(working_output + tf.matmul(working_output, self.working_to_query))
        attach_decisions, _ = self.read_memory(memory_state, attach_queries)
        masked_attach_decisions = attach_decisions * tf.transpose(end)
        decision_output = tf.expand_dims(masked_attach_decisions, 0)
        
        # Update salience memory
        n_updates = tf.reduce_sum(end)
        new_memory_state = self.write_memory(memory_state, masked_attach_decisions, attach_queries, n_updates)
        # Decay salience
        decayed_memory_state = self.decay_salience(new_memory_state)
        
        return (decision_output, inner_output), \
               (new_working_state, decayed_memory_state, new_inner_state)


class EntityLSTMCell(EntityMixin, tf.contrib.rnn.LSTMCell):
    pass


if __name__ == '__main__':
    import argparse, os
    from conll import ConllCorpusReader, DEFAULT_CORPUS_DIR
    from matrix_gen import get_beginning_inside_end, get_attachment_matrix
    from evaluation import get_evaluation
    
    # Command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--print_document_scores", help="Print metrics per document during training", action="store_true")
    parser.add_argument("--print_minibatch_loss", help="Print minibatch (document) loss during training", action="store_true")
    parser.add_argument("--print_dev_loss", help="Print minibatch (document) loss during evaluation on the dev set", action="store_true")
    parser.add_argument("--epochs", help="Number of training epochs", default=20)
    parser.add_argument("--learning_rate", help="Learning rate for training", default=0.001)
    parser.add_argument("--hidden_size", help="Number of hidden units", default=100)
    parser.add_argument("--threshold", help="Threshold value for coreference (between 0 and 1)", default=0.79)
    parser.add_argument("--reg_weight", help="The weight of the regularization term", default=1e-4)
    parser.add_argument("--print_coref_matrices", help="Print gold and predicted coreference matrices", action="store_true")
    parser.add_argument("--additional_features",help="Use vectors containing additional features",action="store_true")
    parser.add_argument("--model_dir", help="Directory for saving models", default="models")
    parser.add_argument("--eval_on_model", help="Path to the model", default="none")
    args = parser.parse_args()
    
    # Data
    conll_reader = ConllCorpusReader.fetch_corpus(cache_dir=os.getcwd())
    
    # Evaluation metrics (TODO not currently used...)
    metrics = ['muc', 'bcub', 'ceafm', 'ceafe', 'blanc', 'lea']
    
    tf.set_random_seed(99)
    
    # Hyperparameters
    
    LEARNING_RATE = float(args.learning_rate)
    EPOCHS = int(args.epochs)
    REGULARIZATION = float(args.reg_weight)
    
    INPUT_SIZE = 362 if args.additional_features else 300
    HIDDEN_SIZE = int(args.hidden_size)
    #MEMORY_SIZE = int(args.hidden_size)
    
    # true MAX_MENTIONS = 522
    MAX_MENTIONS = 30
    
    #EXTRA_SUP_WEIGHT = 0.5

    ### Load data
    
    # Cached embedding matrices
    if args.additional_features:
        train_embs = np.load(os.path.join(DEFAULT_CORPUS_DIR, "training_docs_new.npz"), encoding='latin1')["matrices"]
        dev_embs = np.load(os.path.join(DEFAULT_CORPUS_DIR, "development_docs_new.npz"), encoding='latin1')["matrices"]
        test_embs = np.load(os.path.join(DEFAULT_CORPUS_DIR, "test_docs_new.npz"), encoding='latin1')["matrices"]
    else:
        train_embs = np.load(os.path.join(DEFAULT_CORPUS_DIR, "training_docs.npz"), encoding='latin1')["matrices"]
        dev_embs = np.load(os.path.join(DEFAULT_CORPUS_DIR, "development_docs.npz"), encoding='latin1')["matrices"]
        test_embs = np.load(os.path.join(DEFAULT_CORPUS_DIR, "test_docs.npz"), encoding='latin1')["matrices"]

    # Load the documents, and extract matrices
    # Mention matrix: [num_mentions, num_tokens], with a 1 at the final token of each mention
    # Coref matrix: [num_mentions, num_mentions], with a 1 whenever two mentions are coreferent
    
    train_conll_docs = conll_reader.get_conll_docs("train")
    train_bool_input_matrices = [get_beginning_inside_end(x, MAX_MENTIONS) for x in train_conll_docs]
    train_coref_matrix = [get_attachment_matrix(x) for x in train_conll_docs]

    dev_conll_docs = conll_reader.get_conll_docs("development")
    dev_bool_input_matrices = [get_beginning_inside_end(x, MAX_MENTIONS) for x in dev_conll_docs]
    dev_coref_matrix = [get_attachment_matrix(x) for x in dev_conll_docs]
    
    # Add an extra index (for a "batch size" of 1)
    
    train_embs = [np.expand_dims(x, 0) for x in train_embs]
    dev_embs = [np.expand_dims(x, 0) for x in dev_embs]
    test_embs = [np.expand_dims(x, 0) for x in test_embs]
    train_bool_input_matrices = [[np.expand_dims(x, 0) for x in mats] for mats in train_bool_input_matrices]
    train_coref_matrix = [np.expand_dims(x, 0) for x in train_coref_matrix]
    dev_bool_input_matrices = [[np.expand_dims(x, 0) for x in mats] for mats in dev_bool_input_matrices]
    dev_coref_matrix = [np.expand_dims(x, 0) for x in dev_coref_matrix]
    
    # TODO cache the above
    
    ### Construct the computation graph ###
    
    # Input data
    
    token_to_beginning = tf.placeholder(tf.bool, [1, None, MAX_MENTIONS])  # [batch_size, num_tokens, num_mentions], 1 at beginning of each mention
    token_to_inside = tf.placeholder(tf.bool, [1, None, MAX_MENTIONS]) # [batch_size, num_tokens, num_mentions], 1 at every token inside each mention
    token_to_end = tf.placeholder(tf.bool, [1, None, MAX_MENTIONS])  # [batch_size, num_tokens, num_mentions], 1 at end of each mention
    embeddings = tf.placeholder(tf.float64, [1, None, INPUT_SIZE])  # embeddings [batch_size, num_tokens, input_size] ( = x in lstm_basic)
    gold = tf.placeholder(tf.bool, [1, None, None])  # [batch_size, num_mentions, num_mentions] ( = y in lstm_basic)
    
    gold_float = tf.cast(gold, tf.float64)
    token_to_beginning_float = tf.cast(token_to_beginning, tf.float64)
    token_to_inside_float = tf.cast(token_to_inside, tf.float64)
    token_to_end_float = tf.cast(token_to_end, tf.float64)
    
    # Model
    
    # Apply the Salience LSTM
    cell = EntityLSTMCell(MAX_MENTIONS, HIDDEN_SIZE)
    (decisions, outputs), last_state = tf.nn.dynamic_rnn(cell,
                                                         [embeddings, token_to_beginning_float, token_to_inside_float, token_to_end_float],
                                                         dtype=tf.float64)
    # Get the decision for each mention
    pooled_decisions = tf.reduce_sum(decisions, 1) 
    n_mentions = tf.shape(gold)[-1]
    trimmed_decisions = pooled_decisions[:,:n_mentions,:n_mentions]
    
    x_entropy = - (1/tf.size(gold_float)) * tf.reduce_sum(gold_float * tf.log(trimmed_decisions+(1e-10)) \
                                                          + (1-gold_float) * tf.log(1-trimmed_decisions+(1e-10)))
    
    # TODO give equal total weight to 0s and 1s?
     
    # Add L2 regularization to the cost
    reg = REGULARIZATION*sum([tf.reduce_sum(x**2) for x in tf.trainable_variables()])
    
    # Add an extra supervision signal to predict the input
    #output_to_embeddings = tf.Variable(tf.random_normal((HIDDEN_SIZE, INPUT_SIZE), stddev=0.01, dtype=tf.float64))
    #predicted_embeddings = tf.matmul(tf.squeeze(outputs, 0), output_to_embeddings)
    #difference = predicted_embeddings - tf.squeeze(embeddings, 0)
    #n_tokens = tf.cast(tf.shape(embeddings)[1], tf.float64)
    #extra_sup = EXTRA_SUP_WEIGHT * tf.reduce_sum(difference**2) / n_tokens
    
    cost = x_entropy + reg
    
    # Define the optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
    
    # Start the session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Starting session")
        print("Input size is", INPUT_SIZE)
        
        def print_forward_pass(feed_dict, gold_mat):
            """
            Run forward pass and print outputs
            """
            decision_mat, (_, (memory, salience, index_float), _), this_cost, this_ent = sess.run([trimmed_decisions, last_state, cost, x_entropy], feed_dict=feed_dict)
            index = int(index_float)
            print('decisions:')
            print(decision_mat[0])
            print('gold:')
            print(gold_mat[0])
            print('final salience:')
            final_salience = salience[0,:index]
            top_salience_inds = np.argpartition(final_salience, [-2,-1])[:-3:-1]
            print(final_salience)
            print('most salient memories:')
            print(memory[0, top_salience_inds, :6], "...")
            print('top final salience:')
            print(final_salience[top_salience_inds])
            print('no. mentions:', gold_mat.shape[-1])
            print('cost:', this_cost)
            print('xent:', this_ent)

        for step in range(EPOCHS):
            skipped = 0  # Counter for empty documents, and documents with too many mentions
            # Iterate through documents
            for new_embeddings, new_bool_input, new_gold in zip(train_embs, train_bool_input_matrices, train_coref_matrix):

                #print("===")
                #print("Train docs", new_embeddings.shape)
                #print("Train mention matrix", new_token_to_mention.shape)
                #print("Gold", new_gold.shape)
                
                new_n_mentions = new_gold.shape[-1] 
                
                # Check for empty documents, and documents with too many mentions
                if new_n_mentions == 0 or new_n_mentions > MAX_MENTIONS:
                    skipped += 1
                    continue
                
                # Feed data into computation graph
                feed_dict = {embeddings: new_embeddings,
                             token_to_beginning: new_bool_input[0],
                             token_to_inside: new_bool_input[1],
                             token_to_end: new_bool_input[2],
                             gold: new_gold}
                
                # Train parameters
                sess.run(optimizer, feed_dict=feed_dict)

                print_forward_pass(feed_dict, new_gold)
