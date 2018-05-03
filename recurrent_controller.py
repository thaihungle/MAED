import tensorflow as tf
from controller import BaseController

class RecurrentController(BaseController):

    def network_vars(self):
        print('--define core rnn controller variables--')
        self.lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)
        if not isinstance(self.drop_out_keep, int):
            self.lstm_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_cell,
                                                         output_keep_prob=self.drop_out_keep)
            print('equipe drop out {}'.format(self.drop_out_keep))
        self.state = tf.Variable(tf.zeros([self.batch_size, self.hidden_dim]), trainable=False)
        self.output = tf.Variable(tf.zeros([self.batch_size, self.hidden_dim]), trainable=False)

    def network_op(self, X, state):
        print('--operation rnn controller variables--')
        X = tf.convert_to_tensor(X)
        return self.lstm_cell(X, state) #DO LSTM computation here, return hidden and the state value

    def update_state(self, new_state):
        return tf.group(
            self.output.assign(new_state[0]),
            self.state.assign(new_state[1])
        )

    def get_state(self):
        return self.output, self.state


class StatelessRecurrentController(BaseController):
    def network_vars(self):
        print('--define core rnn stateless controller variables--')
        if self.nlayer==1:
            print('1 layer')
            self.lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)
        else:
            print('{} layers'.format(self.nlayer))
            self.lstm_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(self.hidden_dim) for _ in range(self.nlayer)])
        self.state = self.lstm_cell.zero_state(self.batch_size, tf.float32)

        if not isinstance(self.drop_out_keep, int):
            self.lstm_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_cell,
                                                         output_keep_prob=self.drop_out_keep)
            print('equipe drop out {}'.format(self.drop_out_keep))

    def network_op(self, X, state):
        print('--operation rnn stateless controller variables--')
        X = tf.convert_to_tensor(X)
        return self.lstm_cell(X, state)

    def get_state(self):
        return self.state

    def update_state(self, new_state):
        return tf.no_op()

    def zero_state(self):
        return self.lstm_cell.zero_state(self.batch_size, tf.float32)