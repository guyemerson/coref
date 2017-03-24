import numpy as np
import tensorflow as tf


class SalienceMemoryMixin(tf.contrib.rnn.RNNCell):
    """
    A salience memory extension to be added to any standard RNN cell.
    
    Intended use:
    >>> class CellWithMemory(SalienceMemoryMixin, Cell):
    >>>     pass
    """
    
    def __init__(self, memory_size, max_mentions, *args, **kwargs):
        """
        Set sizes of tensors and initialise weights
        :param memory_size: number of dimensions in each memory vector
        :param max_mentions: maximum number of mentions in an input sequence
        Remaining arguments and keyword arguments are passed to super
        """
        # Initialise the standard RNN
        super().__init__(*args, **kwargs)
        # Sizes of objects
        self.memory_size = memory_size
        self.max_mentions = max_mentions
        self.inner_output_size = super().output_size
        # We also need RNN updates when queries are returned
        # To do this, we need a new set of weights
        # The Method Resolution Order should be:
        # [Child, SalienceMemoryMixin, Parent, RNNCell, object]
        ParentClass = type(self).__mro__[2]
        with tf.variable_scope('secondary_scope'):
            self.secondary_cell = ParentClass(*args, **kwargs)
        # Weights
        # Connection from hidden state to memory
        self.hidden_to_memory = tf.Variable(tf.random_normal((self.inner_output_size, memory_size), stddev=0.01, dtype=tf.float64))
        # Special parameters
        self.salience_decay = tf.constant(0.9, tf.float64) #tf.Variable(, dtype=tf.float64)
        self.new_entity_score = tf.constant(0.5, tf.float64) #tf.Variable(, dtype=tf.float64)
        self.attention_beta = tf.constant(5., tf.float64) #tf.Variable(, dtype=tf.float64)
    
    @property
    def state_size(self):
        """
        Return the shapes of the hidden state, including the memory extension
        """
        return (tf.TensorShape([self.max_mentions, self.memory_size]),  # Memory vectors
                tf.TensorShape([self.max_mentions]),  # Salience of memories
                tf.TensorShape([]),  # Index of next free memory  
                *super().state_size)  # State size of parent RNN cell
    
    @property
    def output_size(self):
        """
        Return the shapes of the output, including memory retrieval decisions
        """
        return (tf.TensorShape([self.max_mentions]),  # Memory retrieval decisions
                super().output_size)  # Output from parent RNN cell
    
    def __call__(self, inputs, state, scope=None):
        """
        Run the RNN
        :param inputs: inputs of shapes ([batch_size, input_size], [batch_size])
        :param state: tuple of state tensors
        :param scope: VariableScope for the created subgraph
        :return: output, new_state
        """
        # Unpack input and state
        embedding, use_memory_float = inputs
        memory, salience, index, *hidden = state
        # Call the parent RNN
        inner_output, new_hidden = super().__call__(embedding, hidden, scope) ### TODO
        
        ### Consult the memory if required
        
        # Cast the bool as a float, and squeeze to a scalar
        use_memory = tf.squeeze(tf.cast(use_memory_float, tf.bool))
        # Use the memory extension only if this bool is true
        # If we don't, leave the state the same, and output zeros as a decision
        new_output, decisions, new_memory, new_salience, new_index, *new_new_hidden = tf.cond(use_memory,
            lambda: self.check_memory(inner_output, memory, salience, index, *new_hidden),
            lambda: (inner_output, tf.zeros((1,self.max_mentions), dtype=tf.float64), memory, salience, index, *new_hidden))

        return (new_output, decisions), (new_memory, new_salience, new_index, *new_new_hidden)
    
    def check_memory(self, inner_output, memory, salience, index_float, *hidden):
        """
        Use the memory extension
        :param inner_output: output from parent RNN
        :param hidden: current hidden state
        :param memory: current memory
        :param salience: current salience
        :param index_float: index of next free memory slot 
        """
        # Cast index from float to int
        index = tf.cast(index_float, tf.int32)
        # Query over memories
        # (what activation function?)
        query = tf.nn.relu(tf.matmul(inner_output, self.hidden_to_memory))
        # Cosine similarity between query and memories
        normed_query = tf.nn.l2_normalize(query, dim=1)
        normed_memory = tf.nn.l2_normalize(memory, dim=2)
        # (squeeze to deal with batch of size 1)
        similarity = tf.matmul(normed_query,  # [1, memory_size]
                               tf.squeeze(normed_memory, 0),  # [1, max_mentions, memory_size]
                               transpose_b=True)
        # TODO dimension size is lost after tf.squeeze...
        weighted_sim = similarity * salience
        # Take attention over these similarities,
        # with a default score for the new entity
        # and no attention for later entities
        # (An alternative attention would involve two decisions:)
        # (whether to use an existing entity, and if so, which  )
        infs = tf.ones((1,self.max_mentions), dtype=tf.float64) * -np.inf
        mask = tf.sequence_mask([index[0]+1], self.max_mentions)
        scores = tf.where(mask, weighted_sim, infs) + tf.one_hot(index, self.max_mentions, self.new_entity_score, dtype=tf.float64)
        attention = tf.nn.softmax(scores * self.attention_beta)  # [1, max_mentions]
        # Add the query vector to the memory
        memory_update = tf.expand_dims(tf.transpose(attention) * query, 0)  # Use broadcasting to do outer product
        new_memory = memory + memory_update
        # Retrieve a memory as a weighted sum, and update the hidden state
        retrieved_memory = tf.matmul(attention, tf.squeeze(new_memory, 0))
        
        # Use the secondary cell (of the type of the parent class) to update the hidden state
        with tf.variable_scope('secondary_scope'):
            new_output, new_hidden = self.secondary_cell.__call__(retrieved_memory, hidden)
        
        # Decay the salience and increase the salience of the activated memory slot(s)
        # (There could be better ways of updating the salience)
        new_salience = tf.maximum(salience * self.salience_decay, attention)
        # Increment the index and cast back to a float
        new_index = index+1
        new_index_float = tf.cast(new_index, tf.float64)
        
        return (new_output, attention, new_memory, new_salience, new_index_float, *new_hidden)


class SalienceLSTMCell(SalienceMemoryMixin, tf.contrib.rnn.LSTMCell):
    pass


if __name__ == '__main__':
    # Hyperparameters
    
    LEARNING_RATE = 0.001
    REGULARIZATION = 0.
    
    INPUT_SIZE = 7
    HIDDEN_SIZE = 5
    MEMORY_SIZE = 4
    MAX_MENTIONS = 5
    
    # Input data
    
    token_to_mention = tf.placeholder(tf.bool, [1, None, None])  # [batch_size, num_tokens, num_mentions]
    embeddings = tf.placeholder(tf.float64, [1, None, INPUT_SIZE])  # embeddings [batch_size, num_tokens, input_size]
    gold = tf.placeholder(tf.bool, [1, None, None])  # [batch_size, num_mentions, num_mentions]
    
    gold_float = tf.cast(gold, tf.float64)
    token_to_mention_float = tf.cast(token_to_mention, tf.float64)
    mention_float = tf.reshape(tf.reduce_sum(token_to_mention_float, 2), (1, -1, 1))  # mention-or-not [batch_size, num_tokens, 1]
    
    # Model
    
    cell = SalienceLSTMCell(MEMORY_SIZE, MAX_MENTIONS, HIDDEN_SIZE)
    (outputs, decisions), last_state = tf.nn.dynamic_rnn(cell, [embeddings, mention_float], dtype=tf.float64)
    
    masked_decisions = tf.matmul(token_to_mention_float, decisions, transpose_a=True)
    
    predictions = tf.matmul(masked_decisions, masked_decisions, transpose_b=True)
    x_entropy =  - (1/tf.size(gold_float)) * tf.reduce_sum(gold_float*tf.log(predictions+(1e-10)) + \
                                                           (1-gold_float)*tf.log(1-predictions+(1e-10)))
    reg = REGULARIZATION*sum([tf.reduce_sum(x**2) for x in tf.trainable_variables()])
    cost = x_entropy + reg
    
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Toy data
        toy_embeddings = np.random.random((1,11,INPUT_SIZE))
        toy_token_to_mention = np.zeros((1,11,3), dtype=np.bool)
        toy_token_to_mention[0,0,0] = 1
        toy_token_to_mention[0,3,1] = 1
        toy_token_to_mention[0,9,2] = 1
        toy_gold = np.eye(3, dtype=np.bool).reshape((1,3,3))
        toy_gold[0,0,1] = 1
        toy_gold[0,1,0] = 1
        
        feed_dict = {embeddings: toy_embeddings,
                     token_to_mention: toy_token_to_mention,
                     gold: toy_gold}
        
        def print_forward_pass():
            decision_mat, final_state, this_cost = sess.run([decisions, last_state, cost], feed_dict=feed_dict)
            print('outputs:')
            print(decision_mat)
            print('final state:')
            for i, s in enumerate(final_state):
                print(i)
                print(s)
            print('cost:')
            print(this_cost)
        
        print_forward_pass()
        
        for _ in range(10):
            sess.run(optimizer, feed_dict=feed_dict)
        
        
        print_forward_pass()