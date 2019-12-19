# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import tensorflow as tf
import matplotlib.gridspec as gridspec
from libs.utils import weight_variable, bias_variable
import matplotlib.pyplot as plt
import numpy as np
from libs import input_data
# from tensorflow.examples.tutorials.mnist import input_data
# import tensorflow_datasets
# 下载mnist数据集
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tensorflow_google", one_hot = True)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # Here "one_hot" is used to one hot labels.


# mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

batch_size = 256
g_dim = 100    # 为输入 noise 的维度大小


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding = 'SAME')


def deconv2d(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides = [1, 2, 2, 1], padding = 'SAME')


x_d = tf.placeholder(tf.float32, shape=[None, 784])
x_g = tf.placeholder(tf.float32, shape=[None, g_dim])
weights = {
    "w_d1": weight_variable([5, 5, 1, 32], "w_d1"),
    "w_d2": weight_variable([5, 5, 32, 64], "w_d2"),
    "w_d3": weight_variable([7 * 7 * 64, 1], "w_d3"),

    "w_g1": weight_variable([g_dim, 4 * 4 * 64], "w_g1"),
    "w_g2": weight_variable([5, 5, 32, 64], "w_g2"),
    "w_g3": weight_variable([5, 5, 16, 32], "w_g3"),
    "w_g4": weight_variable([5, 5, 1, 16], "w_g4")
}

biases = {
    "b_d1": bias_variable([32], "b_d1"),
    "b_d2": bias_variable([64], "b_d2"),
    "b_d3": bias_variable([1], "b_d3"),
    "b_g1": bias_variable([4 * 4 * 64], "b_g1"),
    "b_g2": bias_variable([32], "b_g2"),
    "b_g3": bias_variable([16], "b_g3"),
    "b_g4": bias_variable([1], "b_g4"),
}

var_d = [weights["w_d1"], weights["w_d2"], weights["w_d3"], biases["b_d1"], biases["b_d2"], biases["b_d3"]]
var_g = [weights["w_g1"], weights["w_g2"], weights["w_g3"], weights["w_g4"], biases["b_g1"], biases["b_g2"],
         biases["b_g3"], biases["b_g4"]]


def generator(z):
    # 100 x 1
    h_g1 = tf.nn.relu(tf.add(tf.matmul(z, weights["w_g1"]), biases["b_g1"]))
    # -1 x 4*4*128
    h_g1_reshape = tf.reshape(h_g1, [-1, 4, 4, 64])

    output_shape_g2 = tf.stack([tf.shape(z)[0], 7, 7, 32])
    h_g2 = tf.nn.relu(tf.add(deconv2d(h_g1_reshape, weights["w_g2"], output_shape_g2), biases["b_g2"]))

    output_shape_g3 = tf.stack([tf.shape(z)[0], 14, 14, 16])
    h_g3 = tf.nn.relu(tf.add(deconv2d(h_g2, weights["w_g3"], output_shape_g3), biases["b_g3"]))

    output_shape_g4 = tf.stack([tf.shape(z)[0], 28, 28, 1])
    h_g4 = tf.nn.tanh(tf.add(deconv2d(h_g3, weights["w_g4"], output_shape_g4), biases["b_g4"]))

    return h_g4


def discriminator(x):
    x_reshape = tf.reshape(x, [-1, 28, 28, 1])
    # 28 x 28 x 1
    h_d1 = tf.nn.relu(tf.add(conv2d(x_reshape, weights["w_d1"]), biases["b_d1"]))
    # 14 x 14 x 32
    h_d2 = tf.nn.relu(tf.add(conv2d(h_d1, weights["w_d2"]), biases["b_d2"]))
    # 7 x 7 x 64
    h_d2_reshape = tf.reshape(h_d2, [-1, 7 * 7 * 64])
    h_d3 = tf.nn.sigmoid(tf.add(tf.matmul(h_d2_reshape, weights["w_d3"]), biases["b_d3"]))
    return h_d3


def sample_Z(m, n):  # 确定 noise
    return np.random.uniform(-1., 1., size=[m, n])


g_sample = generator(x_g)
d_real = discriminator(x_d)
d_fake = discriminator(g_sample)

d_loss = -tf.reduce_mean(tf.log(d_real) + tf.log(1. - d_fake))  # -E(logD(x)) - E(log(1 - D(G(z))))
# Optimizatin : E(logD(x)) + E(log( - D(G(z))))

g_loss = -tf.reduce_mean(tf.log(d_fake))


# 只更新 discriminator
d_optimizer = tf.train.AdamOptimizer(0.0001).minimize(d_loss, var_list= var_d)
# 只更新 generator parameters
g_optimizer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list= var_g)


########################   Training   #######################

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap = 'gray')

    plt.show()


sess = tf.InteractiveSession()
init_op = tf.global_variables_initializer()
sess.run(init_op)
for step in range(20001):    # The Batch operation here seems to have some problems.
    batch_x = mnist.train.next_batch(batch_size)[0]
    _, d_loss_train = sess.run([d_optimizer, d_loss], feed_dict={x_d: batch_x, x_g: sample_Z(batch_size, g_dim)})
    _, g_loss_train = sess.run([g_optimizer, g_loss], feed_dict={x_g: sample_Z(batch_size, g_dim)})

    if step <= 1000:
        if step % 100 == 0:
            print("step %d, discriminator loss %.5f" % (step, d_loss_train)),
            print(" generator loss %.5f" % (g_loss_train))
        if step % 1000 == 0:
            g_sample_plot = g_sample.eval(feed_dict = {x_g: sample_Z(16, g_dim)})
            plot(g_sample_plot)
    else:
        if step % 1000 == 0:
            print("step %d, discriminator loss %.5f" % (step, d_loss_train)),
            print(" generator loss %.5f" % (g_loss_train))
        if step % 2000 == 0:
            g_sample_plot = g_sample.eval(feed_dict = {x_g: sample_Z(16, g_dim)})
            plot(g_sample_plot)