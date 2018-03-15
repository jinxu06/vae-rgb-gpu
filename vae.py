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
all_params = tf.trainable_variables()
for i in range(FLAGS.nr_gpu):
    with tf.device('/gpu:%d' % i):
        MSEs[i] = tf.reduce_sum(tf.square(flatten(xs[i])-flatten(x_hats[i])), 1)
        KLDs[i] = - 0.5 * tf.reduce_mean(1 + log_vars[i] - tf.square(locs[i]) - tf.exp(log_vars[i]), axis=-1)
        losses[i] = tf.reduce_mean(MSEs[i] + FLAGS.beta * tf.maximum(FLAGS.lam, KLDs[i]))
        grads[i] = tf.gradients(losses[i], all_params, colocate_gradients_with_ops=True)

with tf.device('/gpu:0'):
    for i in range(1, FLAGS.nr_gpu):
        losses[0] += losses[i]
        for j in range(len(grads[0])):
            grads[0][j] += grads[i][j]

    MSE = tf.concat(MSEs, axis=0)
    KLD = tf.concat(KLDs, axis=0)

    train_step = adam_updates(all_params, grads[0], lr=0.0001)

    loss = losses[0] / FLAGS.nr_gpu

initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

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



if FLAGS.debug:
    train_data = celeba_data.DataLoader(FLAGS.data_dir, 'valid', FLAGS.batch_size*FLAGS.nr_gpu, shuffle=True, size=128)
else:
    train_data = celeba_data.DataLoader(FLAGS.data_dir, 'train', FLAGS.batch_size*FLAGS.nr_gpu, shuffle=True, size=128)
test_data = celeba_data.DataLoader(FLAGS.data_dir, 'valid', FLAGS.batch_size*FLAGS.nr_gpu, shuffle=False, size=128)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    # init
    sess.run(initializer)

    if FLAGS.load_params:
        ckpt_file = FLAGS.save_dir + '/params_' + 'celeba' + '.ckpt'
        print('restoring parameters from', ckpt_file)
        saver.restore(sess, ckpt_file)

    train_mgen = m.RandomRectangleMaskGenerator(128, 128, max_ratio=0.75)
    test_mgen = m.CenterMaskGenerator(128, 128, 0.5)

    max_num_epoch = 1000
    for epoch in range(max_num_epoch):
        tt = time.time()
        ls, mses, klds = [], [], []
        for data in train_data:
            feed_dict = make_feed_dict(data, train_mgen)
            l, mse, kld, _ = sess.run([loss, MSE, KLD, train_step], feed_dict=feed_dict)
            ls.append(l)
            mses.append(mse)
            klds.append(kld)
        train_loss, train_mse, train_kld = np.mean(ls), np.mean(mses), np.mean(klds)

        ls, mses, klds = [], [], []
        for data in test_data:
            feed_dict = make_feed_dict(data)
            l, mse, kld = sess.run([loss, MSE, KLD], feed_dict=feed_dict)
            ls.append(l)
            mses.append(mse)
            klds.append(kld)
        test_loss, test_mse, test_kld = np.mean(ls), np.mean(mses), np.mean(klds)

        print("epoch {0} --------------------- Time {1:.2f}s".format(epoch, time.time()-tt))
        print("train loss:{0:.3f}, train mse:{1:.3f}, train kld:{2:.3f}".format(train_loss, train_mse, train_kld))
        print("test loss:{0:.3f}, test mse:{1:.3f}, test kld:{2:.3f}".format(test_loss, test_mse, test_kld))

        if epoch % FLAGS.save_interval == 0:

            saver.save(sess, FLAGS.save_dir + '/params_' + 'celeba' + '.ckpt')

            data = next(test_data)
            feed_dict = make_feed_dict(data, test_mgen)
            sample_x = sess.run(x_hats, feed_dict=feed_dict)
            sample_x = np.concatenate(sample_x, axis=0)
            test_data.reset()

            img_tile = plotting.img_tile(sample_x[:25], aspect_ratio=1.0, border_color=1.0, stretch=True)
            img = plotting.plot_img(img_tile, title=FLAGS.data_set + ' samples')
            plotting.plt.savefig(os.path.join(FLAGS.save_dir,'%s_vae_sample%d.png' % (FLAGS.data_set, epoch)))
