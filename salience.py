import numpy as np
import tensorflow as tf


class SalienceMemoryRNNCell(tf.contrib.rnn.RNNCell):
    """
    A vanilla RNN cell with a salience memory extension
    """
    
    def __init__(self, input_size, hidden_size, memory_size, max_mentions):
        """
        Set sizes of tensors and initialise weights
        :param input_size: number of dimensions in the input embeddings
        :param hidden_size: number of dimensions in the hidden state
        :param memory_size: number of dimensions in each memory vector
        :param max_mentions: maximum number of mentions in an input sequence
        """
        # Sizes of objects
        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.memory_size = memory_size
        self.max_mentions = max_mentions
        # Weights
        # Connections between layers
        self.input_to_hidden = tf.Variable(tf.random_normal((input_size, hidden_size), stddev=0.01, dtype=tf.float64))
        self.hidden_to_hidden = tf.Variable(tf.random_normal((hidden_size, hidden_size), stddev=0.01, dtype=tf.float64))
        self.hidden_to_memory = tf.Variable(tf.random_normal((hidden_size, memory_size), stddev=0.01, dtype=tf.float64))
        self.hidden_to_hidden_from_memory = tf.Variable(tf.random_normal((hidden_size, hidden_size), stddev=0.01, dtype=tf.float64))
        self.memory_to_hidden = tf.Variable(tf.random_normal((memory_size, hidden_size), stddev=0.01, dtype=tf.float64))
        # Special parameters
        self.salience_decay = tf.constant(0.9, tf.float64) #tf.Variable(, dtype=tf.float64)
        self.new_entity_score = tf.constant(0.5, tf.float64) #tf.Variable(, dtype=tf.float64)
        self.attention_beta = tf.constant(5., tf.float64) #tf.Variable(, dtype=tf.float64)
        # Collect variables
        self.weight_matrices = [self.input_to_hidden, self.hidden_to_hidden, self.hidden_to_memory, self.hidden_to_hidden_from_memory, self.memory_to_hidden]
        self.special_parameters = [self.salience_decay, self.new_entity_score, self.attention_beta]
    
    @property
    def state_size(self):
        """
        Return the shapes of the hidden state, including the memory extension
        """
        return (tf.TensorShape([self.hidden_size]),  # Hidden layer
                tf.TensorShape([self.max_mentions, self.memory_size]),  # Memory vectors
                tf.TensorShape([self.max_mentions]),  # Salience of memories
                tf.TensorShape([]))  # Index of next free memory  
    
    @property
    def output_size(self):
        """
        Return the shapes of the output, including memory retrieval decisions
        """
        return (tf.TensorShape([self.hidden_size]),  # Output from hidden layer
                tf.TensorShape([self.max_mentions]))  # Memory retrieval decisions
    
    def __call__(self, inputs, state, scope=None):
        """
        Run the RNN
        :param inputs: inputs of shapes ([batch_size, input_size], [batch_size])
        :param state: tuple of state tensors
        :param scope: VariableScope for the created subgraph; defaults to 'salience_cell'
        :return: output, new_state
        """
        embedding, use_memory_float = inputs
        hidden, memory, salience, index = state
        new_hidden = tf.nn.relu(tf.matmul(embedding, self.input_to_hidden)
                                + tf.matmul(hidden, self.hidden_to_hidden))
        # Consult the memory if we've reached the end of a mention
        # Cast the bool as a float, and squeeze to a scalar
        use_memory = tf.squeeze(tf.cast(use_memory_float, tf.bool))
        # Use the memory extension only if this bool is true
        # If we don't, leave the state the same, and output zeros as a decision
        new_new_hidden, new_memory, new_salience, new_index, decisions = tf.cond(use_memory,
            lambda: self.check_memory(new_hidden, memory, salience, index),
            lambda: (new_hidden, memory, salience, index, tf.zeros((1,self.max_mentions), dtype=tf.float64)))

        return (new_new_hidden, decisions), (new_new_hidden, new_memory, new_salience, new_index)
    
    def check_memory(self, hidden, memory, salience, index_float):
        """
        Use the memory extension
        :param hidden: current hidden state
        :param memory: current memory
        :param salience: current salience
        :param index_float: index of next free memory slot 
        """
        # Cast index from float to int
        index = tf.cast(index_float, tf.int32)
        # Query over memories
        query = tf.nn.relu(tf.matmul(hidden, self.hidden_to_memory))
        # Cosine similarity between query and memories
        normed_query = tf.nn.l2_normalize(query, dim=1)
        normed_memory = tf.nn.l2_normalize(memory, dim=2)
        # (squeeze to deal with batch of size 1)
        similarity = tf.matmul(normed_query,  # [1, memory_size]
                               tf.squeeze(normed_memory),  # [1, max_mentions, memory_size]
                               transpose_b=True)
        # Take attention over these similarities,
        # with a default score for the new entity
        # and no attention for later entities
        # (An alternative attention would involve two decisions:)
        # (whether to use an existing entity, and if so, which  )
        infs = tf.ones((1,self.max_mentions), dtype=tf.float64) * -np.inf
        mask = tf.sequence_mask([index[0]+1], self.max_mentions)
        scores = tf.where(mask, similarity, infs) + tf.one_hot(index, self.max_mentions, self.new_entity_score, dtype=tf.float64)
        attention = tf.nn.softmax(scores * self.attention_beta)  # [1, max_mentions]
        # Add the query vector to the memory
        memory_update = tf.expand_dims(tf.transpose(attention) * query, 0)  # Use broadcasting to do outer product
        new_memory = memory + memory_update
        # Retrieve a memory as a weighted sum, and update the hidden state
        retrieved_memory = tf.matmul(attention, tf.squeeze(new_memory))
        new_hidden = tf.nn.relu(tf.matmul(retrieved_memory, self.memory_to_hidden)
                                + tf.matmul(hidden, self.hidden_to_hidden_from_memory))
        # Decay the salience and increase the salience of the activated memory slot(s)
        # (There could be better ways of updating the salience)
        new_salience = tf.minimum(salience * self.salience_decay + attention, 1)
        # Increment the index and cast back to a float
        new_index = index+1
        new_index_float = tf.cast(new_index, tf.float64)
        
        return new_hidden, new_memory, new_salience, new_index_float, attention


if __name__ == '__main__':
    # Hyperparameters
    
    LEARNING_RATE = 0.001
    REGULARIZATION = 0.
    
    INPUT_SIZE = 7
    HIDDEN_SIZE = 5
    MEMORY_SIZE = 4
    MAX_MENTIONS = 3
    
    # Input data
    
    token_to_mention = tf.placeholder(tf.bool, [1, None, None])  # [batch_size, num_tokens, num_mentions]
    embeddings = tf.placeholder(tf.float64, [1, None, INPUT_SIZE])  # embeddings [batch_size, num_tokens, input_size]
    gold = tf.placeholder(tf.bool, [1, None, None])  # [batch_size, num_mentions, num_mentions]
    
    gold_float = tf.cast(gold, tf.float64)
    token_to_mention_float = tf.cast(token_to_mention, tf.float64)
    mention_float = tf.reshape(tf.reduce_sum(token_to_mention_float, 2), (1, -1, 1))  # mention-or-not [batch_size, num_tokens, 1]
    
    # Model
    
    cell = SalienceMemoryRNNCell(INPUT_SIZE, HIDDEN_SIZE, MEMORY_SIZE, MAX_MENTIONS)
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