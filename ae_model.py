import tensorflow as tf
import tensorflow.contrib.layers as layers
import math
from pj_model import pj_model

from configs.ae_model import config

class ae_model(pj_model):
    def __init__(self, config, logger=None):
        super(ae_model, self).__init__(config, logger)
        assert len(self.envs) == 1 # only 1 test env

    def initialize(self):
        """
        Assumes the graph has been constructed
        Creates a tf Session and run initializer of variables
        """
        # create tf session
        self.sess = tf.Session()

        # tensorboard stuff
        # self.add_summary()

        # initiliaze all variables
        v_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="q_0/ae_encode/")
        saver = tf.train.Saver({v.name:v for v in v_list})
        saver.restore(self.sess, self.config.ae_model_path)

        init_v_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        init = tf.variables_initializer(list(set(init_v_list) - set(v_list)))
        self.sess.run(init)

        # synchronise q and target_q networks
        #self.sess.run(self.update_target_op)

        # for saving networks weights
        self.saver = tf.train.Saver()

    def get_q_values_op(self, state, scope, index, num_actions):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        # num_actions = self.env.action_space.n
        out = state

        ##############################################################
        """
        TODO: implement a fully connected with no hidden layer (linear
            approximation) using tensorflow. In other words, if your state s
            has a flattened shape of n, and you have m actions, the result of 
            your computation sould be equal to
                W s where W is a matrix of shape m x n

        HINT: you may find tensorflow.contrib.layers useful (imported)
              make sure to understand the use of the scope param

              you can use any other methods from tensorflow
              you are not allowed to import extra packages (like keras,
              lasagne, cafe, etc.)
        """
        ##############################################################
        ################ YOUR CODE HERE - 2-3 lines ################## 

        vs = tf.variable_scope(scope, reuse=False)
        with vs:

            n_filters = [16, 32]
            kernel_sizes = [8, 4]
            strides = [4, 2]
            current_input = out
        
            with tf.variable_scope("ae_encode", reuse=False):
                for layer_i, n_output in enumerate(n_filters):
                    n_input = current_input.get_shape().as_list()[3]
                    W = tf.Variable(tf.random_uniform([kernel_sizes[layer_i], kernel_sizes[layer_i], n_input, n_output],
                        -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)), name="conv_W_" + str(layer_i))
                    b = tf.Variable(tf.zeros([n_output]), name="conv_b_" + str(layer_i))
                    output = tf.nn.relu(tf.add(tf.nn.conv2d(current_input, W, 
                        strides=[1, strides[layer_i], strides[layer_i], 1], padding='SAME'), b))
                    current_input = output

                conv_out = current_input
                conv_out = layers.flatten(inputs=conv_out)

            # out1 = layers.conv2d(inputs=out, num_outputs=16, kernel_size=8, stride=4)
            # out1 = layers.conv2d(inputs=out1, num_outputs=32, kernel_size=4, stride=2)
            # conv_out = layers.flatten(inputs=out1)
                    
            fc_out = layers.fully_connected(inputs=conv_out, num_outputs=256, activation_fn=None)
            fc_out = tf.nn.relu(fc_out) 
            final_out = layers.fully_connected(inputs=fc_out, num_outputs=num_actions, activation_fn=None)

        ##############################################################
        ######################## END YOUR CODE #######################

        return final_out

if __name__ == '__main__':
    # train model
    model = ae_model(config)
    model.run()
