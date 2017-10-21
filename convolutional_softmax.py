import tensorflow as tf
import numpy as np
from numpy import sqrt

class convolutional_softmax(object):
    def __init__(self, input, conv_size, n_conv, n_fully_connected, keep_prob):
        """ Set up the Graph
        arguments
            input:              a tensorflow placeholder for the input tensor
            conv_size:          size of the convolutional filter
            n_conv:             number of convolutional filters
            n_fully_connected:  number of hidden neurons in the hidden layer
            keep_prob           a tensorflow placeholder for the dropout prob
        returns
            nothing
        """

        input_shape = input.get_shape().as_list()

        with tf.variable_scope("conv1") as scope:
            weights = self.cpu_variable("weights", [conv_size, conv_size, input_shape[3], n_conv], tf.truncated_normal_initializer(stddev=0.01))
            biases = self.cpu_variable("biases", [n_conv], tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            activation1 = tf.nn.relu(pre_activation, name=scope.name)

        # No variables to declare => no namespace
        pool1 = tf.nn.max_pool(activation1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")

        with tf.variable_scope("conv2") as scope:
            weights = self.cpu_variable("weights", [conv_size, conv_size, n_conv, n_conv * 2], tf.truncated_normal_initializer(stddev=0.01))
            biases = self.cpu_variable("biases", [n_conv * 2], tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(pool1, weights, [1, 1, 1, 1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            activation2 = tf.nn.relu(pre_activation, name=scope.name)

        shape = activation2.get_shape().as_list()        # a list: [None, 9, 2]
        flat_size = np.prod(shape[1:])            # dim = prod(9,2) = 18
        conv_flat = tf.reshape(activation2, [-1, flat_size])

        with tf.variable_scope("fully_connected"):
            weights = self.cpu_variable("weights", [flat_size, n_fully_connected], tf.truncated_normal_initializer(0.01))
            biases = self.cpu_variable("biases", [n_fully_connected], tf.constant_initializer(0.01))
            linear = tf.matmul(conv_flat, weights)
            pre_activation = tf.nn.bias_add(linear, biases)
            activation3 = tf.nn.relu(pre_activation, name=scope.name)

        dropout = tf.nn.dropout(activation3, keep_prob)

        with tf.variable_scope("softmax_linear"):
            weights = self.cpu_variable("weights", [n_fully_connected, 2], tf.truncated_normal_initializer(sqrt(float(n_fully_connected))))
            biases = self.cpu_variable("biases", [2], tf.constant_initializer(0.01))
            linear = tf.matmul(dropout, weights)
            self.softmax_linear = tf.nn.bias_add(linear, biases, name=scope.name)
            self.prediction = tf.nn.softmax(self.softmax_linear)

    def cpu_variable(self, name, shape, initializer, dtype=tf.float32):
        with tf.device("/cpu:0"):
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
            return var
