# Import the libraries we will need.
import tensorflow as tf
import numpy as np
import input_data
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import os
import scipy.misc
import scipy


# This function performns a leaky relu activation, which is needed for the
# discriminator network.


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


# The below functions are taken from carpdem20's implementation https://github.com/carpedm20/DCGAN-tensorflow
# They allow for saving sample images from the generator to follow progress


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def inverse_transform(images):
    return (images + 1.) / 2.


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j * h:j * h + h, i * w:i * w + w] = image

    return img


def generator(z):

    initializer = tf.truncated_normal_initializer(stddev=0.02)

    zP = slim.fully_connected(z, 4 * 4 * 256, normalizer_fn=slim.batch_norm,
                              activation_fn=tf.nn.relu, scope='g_project', weights_initializer=initializer)
    zCon = tf.reshape(zP, [-1, 4, 4, 256])

    gen1 = slim.convolution2d(
        zCon, num_outputs=128, kernel_size=[3, 3],
        padding="SAME", normalizer_fn=slim.batch_norm,
        activation_fn=tf.nn.relu, scope='g_conv1', weights_initializer=initializer)
    gen1 = tf.depth_to_space(gen1, 2)

    gen2 = slim.convolution2d(
        gen1, num_outputs=64, kernel_size=[3, 3],
        padding="SAME", normalizer_fn=slim.batch_norm,
        activation_fn=tf.nn.relu, scope='g_conv2', weights_initializer=initializer)
    gen2 = tf.depth_to_space(gen2, 2)

    gen3 = slim.convolution2d(
        gen2, num_outputs=32, kernel_size=[3, 3],
        padding="SAME", normalizer_fn=slim.batch_norm,
        activation_fn=tf.nn.relu, scope='g_conv3', weights_initializer=initializer)
    gen3 = tf.depth_to_space(gen3, 2)

    g_out = slim.convolution2d(
        gen3, num_outputs=1, kernel_size=[32, 32], padding="SAME",
        biases_initializer=None, activation_fn=tf.nn.tanh,
        scope='g_out', weights_initializer=initializer)

    return g_out


def discriminator(bottom, cat_list, conts, reuse=False):
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    dis1 = slim.convolution2d(bottom, 32, [3, 3], padding="SAME",
                              biases_initializer=None, activation_fn=lrelu,
                              reuse=reuse, scope='d_conv1', weights_initializer=initializer)
    dis1 = tf.space_to_depth(dis1, 2)

    dis2 = slim.convolution2d(dis1, 64, [3, 3], padding="SAME",
                              normalizer_fn=slim.batch_norm, activation_fn=lrelu,
                              reuse=reuse, scope='d_conv2', weights_initializer=initializer)
    dis2 = tf.space_to_depth(dis2, 2)

    dis3 = slim.convolution2d(dis2, 128, [3, 3], padding="SAME",
                              normalizer_fn=slim.batch_norm, activation_fn=lrelu,
                              reuse=reuse, scope='d_conv3', weights_initializer=initializer)
    dis3 = tf.space_to_depth(dis3, 2)

    dis4 = slim.fully_connected(slim.flatten(dis3), 1024, activation_fn=lrelu,
                                reuse=reuse, scope='d_fc1', weights_initializer=initializer)

    d_out = slim.fully_connected(dis4, 1, activation_fn=tf.nn.sigmoid,
                                 reuse=reuse, scope='d_out', weights_initializer=initializer)

    q_a = slim.fully_connected(dis4, 128, normalizer_fn=slim.batch_norm,
                               reuse=reuse, scope='q_fc1', weights_initializer=initializer)

    # Here we define the unique layers used for the q-network. The number of outputs depends on the number of
    # latent variables we choose to define.
    q_cat_outs = []
    for idx, var in enumerate(cat_list):
        q_outA = slim.fully_connected(q_a, var, activation_fn=tf.nn.softmax,
                                      reuse=reuse, scope='q_out_cat_' + str(idx), weights_initializer=initializer)
        q_cat_outs.append(q_outA)

    q_cont_outs = None
    if conts > 0:
        q_cont_outs = slim.fully_connected(q_a, conts, activation_fn=tf.nn.tanh,
                                           reuse=reuse, scope='q_out_cont_' + str(conts), weights_initializer=initializer)

    return d_out, q_cat_outs, q_cont_outs
