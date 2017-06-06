import tensorflow as tf
import tensorflow.contrib.layers as layers

from core.pj_DQN import pj_DQN

from configs.pj_nature import config
# from configs.q3_nature import config

class pj_model(pj_DQN):
    """
    Implement Fully Connected with Tensorflow
    """
    def add_placeholders_op(self):
        """
        Adds placeholders to the graph

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model
        """
        # this information might be useful
        # here, typically, a state shape is (80, 80, 1)
        # state_shape = list(self.env.observation_space.shape)
        state_shape = [84, 84, 1]

        ##############################################################
        """
        TODO: add placeholders:
              Remember that we stack 4 consecutive frames together, ending up with an input of shape
              (80, 80, 4).
               - self.s: batch of states, type = uint8
                         shape = (batch_size, img height, img width, nchannels x config.state_history)
               - self.a: batch of actions, type = int32
                         shape = (batch_size)
               - self.r: batch of rewards, type = float32
                         shape = (batch_size)
               - self.sp: batch of next states, type = uint8
                         shape = (batch_size, img height, img width, nchannels x config.state_history)
               - self.done_mask: batch of done, type = bool
                         shape = (batch_size)
                         note that this placeholder contains bool = True only if we are done in 
                         the relevant transition
               - self.lr: learning rate, type = float32
        
        (Don't change the variable names!)
        
        HINT: variables from config are accessible with self.config.variable_name
              Also, you may want to use a dynamic dimension for the batch dimension.
              Check the use of None for tensorflow placeholders.

              you can also use the state_shape computed above.
        """
        ##############################################################
        ################YOUR CODE HERE (6-15 lines) ##################

        img_height = state_shape[0]
        img_width = state_shape[1]
        nchannels = state_shape[2]
        self.s = tf.placeholder(tf.uint8, shape=(None, img_height, img_width, nchannels * self.config.state_history), name="s")
        self.a = tf.placeholder(tf.int32, shape=(None,), name="a")
        self.r = tf.placeholder(tf.float32, shape=(None,), name="r")
        self.sp = tf.placeholder(tf.uint8, shape=(None, img_height, img_width, nchannels * self.config.state_history), name="sp")
        self.done_mask = tf.placeholder(tf.bool, shape=(None,), name="done_mask")
        self.lr = tf.placeholder(tf.float32, shape=(), name="lr")

        ##############################################################
        ######################## END YOUR CODE #######################


    def get_q_values_op(self, state, scope, index, num_actions):
	"""
	Return Q values for all actions.
	"""
        out = state

        out1 = layers.conv2d(inputs=out, num_outputs=32, kernel_size=8, stride=4)
        out1 = layers.conv2d(inputs=out1, num_outputs=64, kernel_size=4, stride=2)
        out1 = layers.conv2d(inputs=out1, num_outputs=64, kernel_size=3, stride=1)
        conv_out = layers.flatten(inputs=out1)

        fc_out = layers.fully_connected(inputs=conv_out, num_outputs=512, activation_fn=None)
        fc_out = tf.nn.relu(fc_out) 
        final_out = layers.fully_connected(inputs=fc_out, num_outputs=num_actions, activation_fn=None)

        return final_out

    def add_update_target_op(self, q_scope, target_q_scope):
        """
        update_target_op will be called periodically 
        to copy Q network weights to target Q network

        Remember that in DQN, we maintain two identical Q networks with
        2 different set of weights. In tensorflow, we distinguish them
        with two different scopes. One for the target network, one for the
        regular network. If you're not familiar with the scope mechanism
        in tensorflow, read the docs
        https://www.tensorflow.org/programmers_guide/variable_scope

        Periodically, we need to update all the weights of the Q network 
        and assign them with the values from the regular network. Thus,
        what we need to do is to build a tf op, that, when called, will 
        assign all variables in the target network scope with the values of 
        the corresponding variables of the regular network scope.
    
        Args:
            q_scope: (string) name of the scope of variables for q
            target_q_scope: (string) name of the scope of variables
                        for the target network
        """
        ##############################################################
        """
        TODO: add an operator self.update_target_op that assigns variables
            from target_q_scope with the values of the corresponding var 
            in q_scope

        HINT: you may find the following functions useful:
            - tf.get_collection
            - tf.assign
            - tf.group

        (be sure that you set self.update_target_op)
        """
        ##############################################################
        ################### YOUR CODE HERE - 5-10 lines #############
        
        q_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=q_scope)
        target_q_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_q_scope)
        assigns = [tf.assign(target_q_weights[i], q_weights[i]) for i in xrange(len(q_weights))]
        update_target_op = tf.group(*assigns)
        return update_target_op

        ##############################################################
        ######################## END YOUR CODE #######################


    def add_loss_op(self, q, target_q, num_actions):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
        """
        # you may need this variable
        # num_actions = self.env.action_space.n

        ##############################################################
        """
        TODO: The loss for an example is defined as:
                Q_samp(s) = r if done
                          = r + gamma * max_a' Q_target(s', a')
                loss = (Q_samp(s) - Q(s, a))^2 

              You need to compute the average of the loss over the minibatch
              and store the resulting scalar into self.loss

        HINT: - config variables are accessible through self.config
              - you can access placeholders like self.a (for actions)
                self.r (rewards) or self.done_mask for instance
              - you may find the following functions useful
                    - tf.cast
                    - tf.reduce_max / reduce_sum
                    - tf.one_hot
                    - ...

        (be sure that you set self.loss)
        """
        ##############################################################
        ##################### YOUR CODE HERE - 4-5 lines #############

        q_samp = self.r + self.config.gamma * tf.reduce_max(target_q, axis=1) * (1 - tf.cast(self.done_mask, dtype=tf.float32))
        # loss = tf.reduce_mean(tf.square(q_samp - tf.reduce_sum(q * tf.one_hot(self.a, num_actions), axis=1)))
        delta = 1.0
        residual = tf.abs(q_samp - tf.reduce_sum(q * tf.one_hot(self.a, num_actions), axis=1))
        loss = tf.reduce_mean(tf.where(tf.less(residual, delta), 0.5 * tf.square(residual), delta * residual - 0.5 * tf.square(delta)))
        return loss

        ##############################################################
        ######################## END YOUR CODE #######################


    def add_optimizer_op(self, scope, loss):
        """
        Set self.train_op and self.grad_norm
        """

        ##############################################################
        """
        TODO: 1. get Adam Optimizer (remember that we defined self.lr in the placeholders
                section)
              2. compute grads wrt to variables in scope for self.loss
              3. clip the grads by norm with self.config.clip_val if self.config.grad_clip
                is True
              4. apply the gradients and store the train op in self.train_op
               (sess.run(train_op) must update the variables)
              5. compute the global norm of the gradients and store this scalar
                in self.grad_norm

        HINT: you may find the following functinos useful
            - tf.get_collection
            - optimizer.compute_gradients
            - tf.clip_by_norm
            - optimizer.apply_gradients
            - tf.global_norm
             
             you can access config variable by writing self.config.variable_name

        (be sure that you set self.train_op and self.grad_norm)
        """
        ##############################################################
        #################### YOUR CODE HERE - 8-12 lines #############

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        gs, vs = zip(*optimizer.compute_gradients(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)))
        if self.config.grad_clip is True:
            # gs, _ = tf.clip_by_global_norm(gs, self.config.clip_val)
            gs = [tf.clip_by_norm(g, self.config.clip_val) for g in gs]
        grad_norm = tf.global_norm(gs)
        train_op = optimizer.apply_gradients(zip(gs, vs))
        return train_op, grad_norm
        ##############################################################
        ######################## END YOUR CODE #######################
    

class ProgressiveModel(pj_model):

    def get_q_values_op(self, state, scope, index, num_actions):
        """
        Returns Q values for all actions
	Cross connections from the hidden layers of the previous envs are added.

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
        conv_outs = []
        fc_outs = []
        for i in xrange(0, index + 1):
            if i != index:
                vs = tf.variable_scope("q_" + str(i), reuse=True)
            else:
                vs = tf.variable_scope(scope, reuse=False)

            with vs:
                out1 = layers.conv2d(inputs=out, num_outputs=32, kernel_size=8, stride=4)
                out1 = layers.conv2d(inputs=out1, num_outputs=64, kernel_size=4, stride=2)
                out1 = layers.conv2d(inputs=out1, num_outputs=64, kernel_size=3, stride=1)
                conv_out = layers.flatten(inputs=out1)
                    
                fc_out = layers.fully_connected(inputs=conv_out, num_outputs=512, activation_fn=None)
                for co in conv_outs:
                    fc_out += layers.fully_connected(inputs=co, num_outputs=512, activation_fn=None)
                fc_out = tf.nn.relu(fc_out) 

                if i == index:
                    final_out = layers.fully_connected(inputs=fc_out, num_outputs=num_actions, activation_fn=None)
                    for fco in fc_outs:
                        final_out += layers.fully_connected(inputs=fco, num_outputs=num_actions, activation_fn=None)
                else:
                    conv_outs.append(conv_out)
                    fc_outs.append(fc_out)    

        # with tf.variable_scope(scope, reuse=False):
        #     out = layers.conv2d(inputs=out, num_outputs=32, kernel_size=8, stride=4)
        #     out = layers.conv2d(inputs=out, num_outputs=64, kernel_size=4, stride=2)
        #     out = layers.conv2d(inputs=out, num_outputs=64, kernel_size=3, stride=1)
        #     out = layers.flatten(inputs=out)
        #     out = layers.fully_connected(inputs=out, num_outputs=512)
        #     out = layers.fully_connected(inputs=out, num_outputs=num_actions, activation_fn=None)
        
        # if prev_scope is None:
        #     with tf.variable_scope(scope, reuse=False):
        #         out = layers.conv2d(inputs=out, num_outputs=32, kernel_size=8, stride=4)
        #         out = layers.conv2d(inputs=out, num_outputs=64, kernel_size=4, stride=2)
        #         out = layers.conv2d(inputs=out, num_outputs=64, kernel_size=3, stride=1)
        #         out = layers.flatten(inputs=out)
        #         out = layers.fully_connected(inputs=out, num_outputs=512)
        #         out = layers.fully_connected(inputs=out, num_outputs=num_actions, activation_fn=None)
        # else:
        #     with tf.variable_scope(prev_scope, reuse=True):
        #         out1 = layers.conv2d(inputs=out, num_outputs=32, kernel_size=8, stride=4)
        #         out1 = layers.conv2d(inputs=out1, num_outputs=64, kernel_size=4, stride=2)
        #         out1 = layers.conv2d(inputs=out1, num_outputs=64, kernel_size=3, stride=1)
        #         out1 = layers.flatten(inputs=out1)
        #         out2 = layers.fully_connected(inputs=out1, num_outputs=512)
        #     with tf.variable_scope(scope, reuse=False):
        #         out = layers.conv2d(inputs=out, num_outputs=32, kernel_size=8, stride=4)
        #         out = layers.conv2d(inputs=out, num_outputs=64, kernel_size=4, stride=2)
        #         out = layers.conv2d(inputs=out, num_outputs=64, kernel_size=3, stride=1)
        #         out = layers.flatten(inputs=out)

        #         num_out1 = out1.get_shape().as_list()[1]
        #         num_out2 = out2.get_shape().as_list()[1]
        #         out1 = layers.fully_connected(inputs=out1, num_outputs=num_out1)
        #         out2 = layers.fully_connected(inputs=out2, num_outputs=num_out2)

        #         out = tf.nn.relu(layers.fully_connected(inputs=out, num_outputs=512, activation_fn=None) + \
        #             layers.fully_connected(inputs=out1, num_outputs=512, activation_fn=None))

        #         out = layers.fully_connected(inputs=out, num_outputs=num_actions, activation_fn=None) + \
        #             layers.fully_connected(inputs=out2, num_outputs=num_actions, activation_fn=None)

        ##############################################################
        ######################## END YOUR CODE #######################

        return final_out




if __name__ == '__main__':
    # train model
    #model = pj_model(config)
    model = ProgressiveModel(config)
    model.run()
