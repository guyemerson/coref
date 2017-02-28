import numpy as np
import tensorflow as tf


class SalienceMemoryRNNCell(tf.contrib.rnn.RNNCell):
    
    def __init__(self, input_size, hidden_size, memory_size, max_mentions):
        # Sizes of objects
        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.memory_size = memory_size
        self.max_mentions = max_mentions
        # Weights
        self.input_to_hidden = tf.Variable(tf.random_normal((input_size, hidden_size)))
        self.hidden_to_hidden = tf.Variable(tf.random_normal((hidden_size, hidden_size)))
        self.hidden_to_memory = tf.Variable(tf.random_normal((hidden_size, memory_size)))
    
    @property
    def state_size(self):
        return (tf.TensorShape([self.hidden_size]),  # Hidden layer
                tf.TensorShape([self.max_mentions, self.memory_size]),  # Memory vectors
                tf.TensorShape([self.max_mentions]),  # Salience of memories
                tf.TensorShape([]),  # Index of next free memory
                tf.TensorShape([self.max_mentions, self.max_mentions]))  # Record of memory retrieval decisions (for convenience)  
    
    @property
    def output_size(self):
        return self.hidden_size
    
    def __call__(self, inputs, state, scope=None):
        """
        Run the RNN
        :param inputs: inputs of shapes ([batch_size, input_size], [batch_size])
        :param state: tuple of state tensors
        :param scope: VariableScope for the created subgraph; defaults to 'salience_cell'
        :return: output, new_state
        """
        with tf.variable_scope(scope or 'salience_cell'):  # This is best practice?
            embedding, use_memory_float = inputs
            hidden, memory, salience, index, decisions = state
            new_hidden = tf.nn.relu(tf.matmul(embedding, self.input_to_hidden)
                                    + tf.matmul(hidden, self.hidden_to_hidden))
            # Consult the memory if we've reached the end of a mention
            # Cast the bool as a float, and squeeze to a scalar
            use_memory = tf.squeeze(tf.cast(use_memory_float, tf.bool))
            new_new_hidden, new_memory, new_salience, new_index, new_decisions = tf.cond(use_memory,
                lambda: self.check_memory(new_hidden, memory, salience, index, decisions),
                lambda: (new_hidden, memory, salience, index, decisions))

            return new_new_hidden, (new_new_hidden, new_memory, new_salience, new_index, new_decisions)
    
    def check_memory(self, hidden, memory, salience, index_float, decisions):
        
        '''
        TODO
        query = fn(hidden)
        attention_over_memories = fn2(query, memory, salience)
        new_memory = memory + "tf.outer"(attention_over_memories, query)
        returned_vector = tf.matmul(attention_over_memories, new_memory)
        new_hidden = fn3(hidden, returned_vector)
        '''
        index = tf.cast(index_float, tf.int32)
        
        new_memory = memory
        
        new_salience = salience * 0.9 + tf.one_hot(index, self.max_mentions)
        
        new_index = index+1
        new_index_float = tf.cast(new_index, tf.float32)
        
        new_decisions = decisions
        
        new_hidden = hidden
        
        return new_hidden, new_memory, new_salience, new_index_float, new_decisions


if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [None, None, 7])  # embeddings [batch_size, num_tokens, input_size]
    m = tf.placeholder(tf.bool, [None, None, 1])  # mention-or-not [batch_size, num_tokens, 1]
    cell = SalienceMemoryRNNCell(7, 5, 4, 3)
    m_float = tf.cast(m, tf.float32)
    outputs, states = tf.nn.dynamic_rnn(cell, [x, m_float], dtype=tf.float32)  # [batch_size, num_tokens, output_size], [batch_size, num_tokens, state_size]
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        inputs = [np.arange(28).reshape((1,4,7)),
                  np.array([0,1,1,1], dtype='bool').reshape((1,4,1))]
        all_outputs, final_state = sess.run([outputs, states],
                                            feed_dict={x: inputs[0],
                                                       m: inputs[1]})
        print('outputs:')
        print(all_outputs)
        print('states:')
        for i, s in enumerate(final_state):
            print(i)
            print(s)