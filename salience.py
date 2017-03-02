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
        self.input_to_hidden = tf.Variable(tf.random_normal((input_size, hidden_size)))
        self.hidden_to_hidden = tf.Variable(tf.random_normal((hidden_size, hidden_size)))
        self.hidden_to_memory = tf.Variable(tf.random_normal((hidden_size, memory_size)))
        self.hidden_to_hidden_from_memory = tf.Variable(tf.random_normal((hidden_size, hidden_size)))
        self.memory_to_hidden = tf.Variable(tf.random_normal((memory_size, hidden_size)))
        # Special parameters
        self.salience_decay = tf.Variable(0.9)
        self.new_entity_score = tf.Variable(0.5)
        self.attention_beta = tf.Variable(5.)
    
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
            lambda: (new_hidden, memory, salience, index, tf.zeros((1,self.max_mentions))))

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
        infs = tf.ones((1,self.max_mentions)) * -np.inf
        mask = tf.sequence_mask([index[0]+1], self.max_mentions)
        scores = tf.where(mask, similarity, infs) + tf.one_hot(index, self.max_mentions, self.new_entity_score)
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
        new_index_float = tf.cast(new_index, tf.float32)
        
        return new_hidden, new_memory, new_salience, new_index_float, attention


if __name__ == '__main__':
    # Test forward pass with toy data
    x = tf.placeholder(tf.float32, [None, None, 7])  # embeddings [batch_size, num_tokens, input_size]
    m = tf.placeholder(tf.bool, [None, None, 1])  # mention-or-not [batch_size, num_tokens, 1]
    cell = SalienceMemoryRNNCell(7, 5, 4, 3)
    m_float = tf.cast(m, tf.float32)
    outputs, states = tf.nn.dynamic_rnn(cell, [x, m_float], dtype=tf.float32)  # [batch_size, num_tokens, output_size], [batch_size, num_tokens, state_size]
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        inputs = [np.random.random((1,11,7)),
                  np.array([0,1,0,0,0,0,0,0,1,0,1], dtype='bool').reshape((1,11,1))]
        all_outputs, final_state = sess.run([outputs, states],
                                            feed_dict={x: inputs[0],
                                                       m: inputs[1]})
        print('outputs:')
        print(all_outputs)
        print('final state:')
        for i, s in enumerate(final_state):
            print(i)
            print(s)