import numpy as np
import os
import tensorflow as tf
import time
import data.celeba_data as celeba_data
from tensorflow.contrib.framework.python.ops import arg_scope
import pixel_cnn_pp.nn as nn
from utils import plotting
from pixel_cnn_pp.nn import adam_updates
import utils.mask as m


tf.flags.DEFINE_integer("z_dim", default_value=100, docstring="latent dimension")
tf.flags.DEFINE_integer("batch_size", default_value=25, docstring="")
tf.flags.DEFINE_integer("nr_gpu", default_value=1, docstring="number of GPUs")
tf.flags.DEFINE_integer("save_interval", default_value=5, docstring="")
tf.flags.DEFINE_float("lam", default_value=1., docstring="")
tf.flags.DEFINE_float("beta", default_value=1., docstring="")
tf.flags.DEFINE_string("data_dir", default_value="/data/ziz/not-backed-up/jxu/CelebA", docstring="")
tf.flags.DEFINE_string("save_dir", default_value="/data/ziz/jxu/models/vae-test", docstring="")
tf.flags.DEFINE_string("data_set", default_value="celeba128", docstring="")
tf.flags.DEFINE_boolean("load_params", default_value=False, docstring="load_parameters from save_dir?")
tf.flags.DEFINE_boolean("debug", default_value=False, docstring="is debugging?")

FLAGS = tf.flags.FLAGS

kernel_initializer = None #tf.random_normal_initializer()

def generative_network(z):
    with tf.variable_scope("generative_network"):
        net = tf.reshape(z, [-1, 1, 1, FLAGS.z_dim])

        net = tf.layers.conv2d_transpose(net, 1024, 4, strides=1, padding='VALID', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 4x4
        net = tf.layers.conv2d_transpose(net, 512, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 8x8
        net = tf.layers.conv2d_transpose(net, 256, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 16x16
        net = tf.layers.conv2d_transpose(net, 128, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 32x32
        net = tf.layers.conv2d_transpose(net, 64, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 64x64
        net = tf.layers.conv2d_transpose(net, 32, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 128x128
        net = tf.layers.conv2d_transpose(net, 3, 1, strides=1, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.nn.sigmoid(net)
    return net

def inference_network(x):
    with tf.variable_scope("inference_network"):
        net = tf.reshape(x, [-1, 128, 128, 3]) # 128x128x3
        net = tf.layers.conv2d(net, 32, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 64x64
        net = tf.layers.conv2d(net, 64, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 32x32
        net = tf.layers.conv2d(net, 128, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 16x16
        net = tf.layers.conv2d(net, 256, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 8x8
        net = tf.layers.conv2d(net, 512, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 4x4
        net = tf.layers.conv2d(net, 1024, 4, strides=1, padding='VALID', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 1x1
        net = tf.reshape(net, [-1, 1024])
        net = tf.layers.dense(net, FLAGS.z_dim * 2, activation=None, kernel_initializer=kernel_initializer)
        loc = net[:, :FLAGS.z_dim]
        log_var = net[:, FLAGS.z_dim:]
    return loc, log_var

def sample_z(loc, log_var):
    with tf.variable_scope("sample_z"):
        scale = tf.sqrt(tf.exp(log_var))
        dist = tf.distributions.Normal(loc=loc, scale=scale)
        z = dist.sample()
        return z

def vae_model(x, z_dim):
    loc, log_var = inference_network(x)
    z = sample_z(loc, log_var)
    x_hat = generative_network(z)
    return loc, log_var, z, x_hat


model_opt = {"z_dim": FLAGS.z_dim}
model = tf.make_template('vae', vae_model)

xs = [tf.placeholder(tf.float32, shape=(None, 128, 128, 3)) for i in range(FLAGS.nr_gpu)]
ms = [tf.placeholder_with_default(np.ones((FLAGS.batch_size, 128, 128), dtype=np.float32), shape=(None, 128, 128)) for i in range(FLAGS.nr_gpu)]
mxs = tf.multiply(xs, tf.stack([ms for k in range(3)], axis=-1))


locs = [None for i in range(FLAGS.nr_gpu)]
log_vars = [None for i in range(FLAGS.nr_gpu)]
zs = [None for i in range(FLAGS.nr_gpu)]
x_hats = [None for i in range(FLAGS.nr_gpu)]
MSEs = [None for i in range(FLAGS.nr_gpu)]
KLDs = [None for i in range(FLAGS.nr_gpu)]
losses = [None for i in range(FLAGS.nr_gpu)]
grads = [None for i in range(FLAGS.nr_gpu)]


flatten = tf.contrib.layers.flatten

for i in range(FLAGS.nr_gpu):
    with tf.device('/gpu:%d' % i):
        locs[i], log_vars[i], zs[i], x_hats[i] = model(mxs[i], **model_opt)

saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vae'))

def make_feed_dict(data, mgen=None):
    data = np.cast[np.float32](data/255.)
    ds = np.split(data, FLAGS.nr_gpu)
    for i in range(FLAGS.nr_gpu):
        feed_dict = { xs[i]:ds[i] for i in range(FLAGS.nr_gpu) }
    if mgen is not None:
        masks = mgen.gen(data.shape[0])
        masks = np.split(masks, FLAGS.nr_gpu)
        for i in range(FLAGS.nr_gpu):
            feed_dict.update({ ms[i]:masks[i] for i in range(FLAGS.nr_gpu) })
    return feed_dict

def load_vae(sess, saver):

    ckpt_file = FLAGS.save_dir + '/params_' + 'celeba' + '.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, ckpt_file)
