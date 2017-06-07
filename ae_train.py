import os
import tensorflow as tf
import numpy as np
import math
from configs.ae_train import config
from utils.general import get_logger, Progbar
import matplotlib.pyplot as plt

def autoencoder():
    """Build a deep denoising autoencoder w/ tied weights.
    Parameters
    ----------
    input_shape : list, optional
        Description
    n_filters : list, optional
        Description
    filter_sizes : list, optional
        Description
    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training
    Raises
    ------
    ValueError
        Description
    """ # %%
    # input to the network
    state_shape = [84, 84, 1]
    img_height = state_shape[0]
    img_width = state_shape[1]
    nchannels = state_shape[2]
    s = tf.placeholder(tf.uint8, shape=(None, img_height, img_width, nchannels * config.state_history), name="s")

    state = tf.cast(s, tf.float32)
    state /= 255

    n_filters = [16, 32]
    kernel_sizes = [8, 4]
    strides = [4, 2]
    encoder = []
    shapes = []
    current_input = state
    with tf.variable_scope(config.q_scope, reuse=False):
        with tf.variable_scope("ae_encode", reuse=False):
            for layer_i, n_output in enumerate(n_filters):
                n_input = current_input.get_shape().as_list()[3]
                shapes.append(current_input.get_shape().as_list())
                W = tf.Variable(tf.random_uniform([kernel_sizes[layer_i], kernel_sizes[layer_i], n_input, n_output],
                    -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)), name="conv_W_" + str(layer_i))
                b = tf.Variable(tf.zeros([n_output]), name="conv_b_" + str(layer_i))
                encoder.append(W)
                output = tf.nn.relu(tf.add(tf.nn.conv2d(current_input, W, 
                    strides=[1, strides[layer_i], strides[layer_i], 1], padding='SAME'), b))
                current_input = output

    # %%
    # store the latent representation
    h = current_input
    r_encoder = encoder[::-1]
    r_shapes = shapes[::-1]
    r_strides = strides[::-1]

    # %%
    # Build the decoder using the same weights
    with tf.variable_scope("ae_decode", reuse=False):
        for layer_i, shape in enumerate(r_shapes):
            W = r_encoder[layer_i]
            b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
            output = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(current_input, W, tf.stack([tf.shape(s)[0], shape[1], shape[2], shape[3]]),
                strides=[1, r_strides[layer_i], r_strides[layer_i], 1], padding='SAME'), b))
            current_input = output

    # now have the reconstruction through the network
    y = current_input
    # cost function measures pixel-wise difference
    loss = tf.reduce_mean(tf.square(y - state))

    return s, h, loss, y

def add_optimizer_op(loss):
    optimizer = tf.train.AdamOptimizer(learning_rate=config.lr)
    gs, vs = zip(*optimizer.compute_gradients(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
    if config.grad_clip is True:
        # gs, _ = tf.clip_by_global_norm(gs, config.clip_val)
        gs = [tf.clip_by_norm(g, config.clip_val) if g is not None else None for g in gs]
    grad_norm = tf.global_norm(gs)
    train_op = optimizer.apply_gradients(zip(gs, vs))
    return train_op, grad_norm

def minibatches(data, batch_size):
    data_size = len(data)
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    for batch_start in np.arange(0, data_size, batch_size):
        batch_indices = indices[batch_start : batch_start + batch_size]
        yield np.concatenate([data[index] for index in batch_indices])

def load_data():
    #data = np.load(config.input_path + "frames_0_5.npz")
    data = {}
    for i in xrange(100000):
        data[i] = np.random.randint(0,high=255,size=(84,84,4),dtype=np.uint8)

    # data = np.load('mat.npz')
    train_data = []
    eval_data = []
    for key in data:
        if np.random.random() < 0.1:
            eval_data.append(np.expand_dims(data[key], axis=0))
        else:
            train_data.append(np.expand_dims(data[key], axis=0))
    return np.array(train_data), np.array(eval_data)

def save(sess):
    if not os.path.exists(config.model_output):
        os.makedirs(config.model_output)

    v_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=config.q_scope)
    saver = tf.train.Saver({v.name:v for v in v_list})
    saver.save(sess, config.model_output + "ae.ckpt")

def train():
    s, _, loss, y = autoencoder()
    train_op, grad_norm = add_optimizer_op(loss)

    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)
    logger = get_logger(config.log_path)
    
    train_data, eval_data = load_data() 

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in xrange(0, config.epoch_num):
        # each epoch

        #train
        prog = Progbar(target=1 + int(len(train_data) / config.batch_size)) 
        step = 1
        for batch in minibatches(train_data, config.batch_size):
            loss_eval, grad_norm_eval, y_train, _ = sess.run([loss, grad_norm, y, train_op], feed_dict={s: batch})
            prog.update(step, [("train loss", loss_eval), ("grad norm", grad_norm_eval)])
            step += 1
	plt.imshow(y_train[0,:,:,0])

        #eval
        #prog = Progbar(target=1 + int(len(eval_data) / config.batch_size)) 
        #step = 1
        #losses = []
        #for batch in minibatches(eval_data, config.batch_size):
        #    loss_eval = sess.run(loss, feed_dict={s: batch})
        #    prog.update(step, [("eval loss", loss_eval)])
        #    losses.append(loss_eval)
        #    step += 1
        #avg_loss = np.mean(losses)
        #sigma_loss = np.sqrt(np.var(losses) / len(losses))
        #print ""
        #msg = "Average loss: {:04.2f} +/- {:04.2f}".format(avg_loss, sigma_loss)
        #logger.info(msg)

        save(sess)

if __name__ == '__main__':
    train()
