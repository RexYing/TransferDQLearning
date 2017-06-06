import os
import numpy as np
import tensorflow as tf
import time
import copy

from deep_q_learning import DQN
from q1_schedule import LinearExploration, LinearSchedule
import gym
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv, wrap_dqn
from utils.general import get_logger, export_plot


class pj_DQN(DQN):
    def __init__(self, config, logger=None):
        """
        Initialize Q Network and env

        Args:
            config: class with hyperparameters
            logger: logger instance from logging module
        """
        # directory for training outputs
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)
            
        # store hyper params
        self.config = config
        self.logger = logger
        if logger is None:
            self.logger = get_logger(config.log_path)
        envs = []
        for env_name in config.env_names: 
            env = gym.make(env_name)
            env = wrap_dqn(env)
            env = PreproWrapper(env, prepro=greyscale, shape=(84, 84, 1), 
                overwrite_render=config.overwrite_render)
            envs.append(env)
        self.envs = envs

        # build model
        self.build()

    """
    Abstract class for Deep Q Learning
    """
    def add_placeholders_op(self):
        raise NotImplementedError


    def get_q_values_op(self, scope, reuse=False):
        """
        set Q values, of shape = (batch_size, num_actions)
        """
        raise NotImplementedError


    def add_update_target_op(self, q_scope, target_q_scope):
        """
        Update_target_op will be called periodically 
        to copy Q network to target Q network
    
        Args:
            q_scope: name of the scope of variables for q
            target_q_scope: name of the scope of variables for the target
                network
        """
        raise NotImplementedError


    def add_loss_op(self, q, target_q):
        """
        Set (Q_target - Q)^2
        """
        raise NotImplementedError


    def add_optimizer_op(self, scope):
        """
        Set training op wrt to loss for variable in scope
        """
        raise NotImplementedError


    def process_state(self, state):
        """
        Processing of state

        State placeholders are tf.uint8 for fast transfer to GPU
        Need to cast it to float32 for the rest of the tf graph.

        Args:
            state: node of tf graph of shape = (batch_size, height, width, nchannels)
                    of type tf.uint8.
                    if , values are between 0 and 255 -> 0 and 1
        """
        state = tf.cast(state, tf.float32)
        state /= self.config.high

        return state


    def build(self):
        """
        Build model by adding all necessary variables
        """
        # add placeholders
        self.add_placeholders_op()

        s = self.process_state(self.s)
        sp = self.process_state(self.sp)
        
        self.ops = []
        for index, env in enumerate(self.envs):
            q_scope = "q_" + str(index)
            target_q_scope = "target_q_" + str(index)

            q = self.get_q_values_op(s, q_scope, index, env.action_space.n)
            target_q = self.get_q_values_op(sp, target_q_scope, index, env.action_space.n)
            update_target_op = self.add_update_target_op(q_scope, target_q_scope)
            loss = self.add_loss_op(q, target_q, env.action_space.n)
            train_op, grad_norm = self.add_optimizer_op(q_scope, loss)

            self.ops.append((q, target_q, update_target_op, loss, train_op, grad_norm))

        # self.q_1 = self.get_q_values_op(s, "q_1")
        # self.target_q_1 = self.get_q_values_op(sp, "target_q_1")
        # self.update_target_op_1 = self.add_update_target_op("q_1", "target_q_1")
        # self.loss_1 = self.add_loss_op(self.q_1, self.target_q_1)
        # self.train_op_1, self.grad_norm_1 = self.add_optimizer_op("q_1", self.loss_1)

        # self.q_2 = self.get_q_values_op(s, "q_2", prev_scope="q_1")
        # self.target_q_2 = self.get_q_values_op(sp, "target_q_2", prev_scope="q_1")
        # self.update_target_op_2 = self.add_update_target_op("q_2", "target_q_2")
        # self.loss_2 = self.add_loss_op(self.q_2, self.target_q_2)
        # self.train_op_2, self.grad_norm_2 = self.add_optimizer_op("q_2", self.loss_2)


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
        init_v_list = []
        for i in xrange(self.config.start_index, len(self.envs)):
            q_scope = "q_" + str(i)
            target_q_scope = "target_q_" + str(i)
            init_v_list += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=q_scope)
            init_v_list += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_q_scope)
            init = tf.variables_initializer(init_v_list)
        self.sess.run(init)

        for i in xrange(0, self.config.start_index):
            q_scope = "q_" + str(i)
            target_q_scope = "target_q_" + str(i)
            v_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=q_scope) + \
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_q_scope)
            saver = tf.train.Saver({v.name:v for v in v_list})
            saver.restore(self.sess, self.config.model_output + "model_" + str(i) + ".ckpt")

        # synchronise q and target_q networks
        #self.sess.run(self.update_target_op)

        # for saving networks weights
        self.saver = tf.train.Saver()

       
    def add_summary(self):
        """
        Tensorboard stuff
        """
        # extra placeholders to log stuff from python
        self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
        self.max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
        self.std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")

        self.avg_q_placeholder  = tf.placeholder(tf.float32, shape=(), name="avg_q")
        self.max_q_placeholder  = tf.placeholder(tf.float32, shape=(), name="max_q")
        self.std_q_placeholder  = tf.placeholder(tf.float32, shape=(), name="std_q")

        self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")

        # add placeholders from the graph
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("grads norm", self.grad_norm)

        # extra summaries from python -> placeholders
        tf.summary.scalar("Avg Reward", self.avg_reward_placeholder)
        tf.summary.scalar("Max Reward", self.max_reward_placeholder)
        tf.summary.scalar("Std Reward", self.std_reward_placeholder)

        tf.summary.scalar("Avg Q", self.avg_q_placeholder)
        tf.summary.scalar("Max Q", self.max_q_placeholder)
        tf.summary.scalar("Std Q", self.std_q_placeholder)

        tf.summary.scalar("Eval Reward", self.eval_reward_placeholder)
            
        # logging
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.output_path, 
                                                self.sess.graph)



    def save(self):
        """
        Saves session
        """
        if not os.path.exists(self.config.model_output):
            os.makedirs(self.config.model_output)

        q_scope = "q_" + str(self.index)
        target_q_scope = "target_q_" + str(self.index)
        v_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=q_scope) + \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_q_scope)
        saver = tf.train.Saver({v.name:v for v in v_list})
        saver.save(self.sess, self.config.model_output + "model_" + str(self.index) + ".ckpt")
        # self.saver.save(self.sess, self.config.model_output + "model_" + str(self.index) + ".ckpt")


    def get_best_action(self, state):
        """
        Return best action

        Args:
            state: 4 consecutive observations from gym
        Returns:
            action: (int)
            action_values: (np array) q values for all actions
        """
        action_values = self.sess.run(self.q, feed_dict={self.s: [state]})[0]
        return np.argmax(action_values), action_values


    def update_step(self, t, replay_buffer, lr):
        """
        Performs an update of parameters by sampling from replay_buffer

        Args:
            t: number of iteration (episode and move)
            replay_buffer: ReplayBuffer instance .sample() gives batches
            lr: (float) learning rate
        Returns:
            loss: (Q - Q_target)^2
        """

        s_batch, a_batch, r_batch, sp_batch, done_mask_batch = replay_buffer.sample(
            self.config.batch_size)


        fd = {
            # inputs
            self.s: s_batch,
            self.a: a_batch,
            self.r: r_batch,
            self.sp: sp_batch, 
            self.done_mask: done_mask_batch,
            self.lr: lr, 
            # extra info
            # self.avg_reward_placeholder: self.avg_reward, 
            # self.max_reward_placeholder: self.max_reward, 
            # self.std_reward_placeholder: self.std_reward, 
            # self.avg_q_placeholder: self.avg_q, 
            # self.max_q_placeholder: self.max_q, 
            # self.std_q_placeholder: self.std_q, 
            # self.eval_reward_placeholder: self.eval_reward, 
        }

        # loss_eval, grad_norm_eval, summary, _ = self.sess.run([self.loss, self.grad_norm, 
        #                                          self.merged, self.train_op], feed_dict=fd)

        loss_eval, grad_norm_eval, _ = self.sess.run([self.loss, self.grad_norm, self.train_op], feed_dict=fd)

        # tensorboard stuff
        # self.file_writer.add_summary(summary, t)
        
        return loss_eval, grad_norm_eval


    def update_target_params(self):
        """
        Update parametes of Q' with parameters of Q
        """
        self.sess.run(self.update_target_op)

    def run(self):
        """
        Apply procedures of training for a QN

        Args:
            exp_schedule: exploration strategy for epsilon
            lr_schedule: schedule for learning rate
        """
        # initialize
        self.initialize()

        config = self.config
        for index, env in enumerate(self.envs):
            if index < self.config.start_index:
                continue
            exp_schedule = LinearExploration(env, config.eps_begin, 
                config.eps_end, config.eps_nsteps)

            lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
                config.lr_nsteps)

            self.q, self.target_q, self.update_target_op, self.loss, self.train_op, self.grad_norm = self.ops[index]
            self.sess.run(self.update_target_op)
            
            # important 
            self.env = env
            self.index = index

            # record one game at the beginning
            if self.config.record:
                self.record()
            self.train(exp_schedule, lr_schedule)

    def export_score(self, scores_eval):
        export_plot(scores_eval, "Scores", self.config.plot_output + "scores_" + str(self.index) + ".png")

    def record(self):
        """
        Re create an env and record a video for one episode
        """
        env = gym.make(self.config.env_names[self.index])
        env = gym.wrappers.Monitor(env, self.config.record_path, video_callable=lambda x: True, resume=True)
        env = wrap_dqn(env)
        env = PreproWrapper(env, prepro=greyscale, shape=(84, 84, 1),
            overwrite_render=self.config.overwrite_render)
        self.evaluate(env, 1)
