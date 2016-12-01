import os
import numpy as np
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import tqdm  # making loops prettier
import h5py  # for reading our dataset
from os import listdir
from os.path import isfile, join
import scipy.misc
import tensorflow.contrib.slim as slim
import vgg


def generator(z, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):

        initializer = tf.truncated_normal_initializer(stddev=0.02)

        zP = slim.fully_connected(z, 4 * 4 * 256, normalizer_fn=slim.batch_norm,
                                  activation_fn=tf.nn.relu, scope='g_project', weights_initializer=initializer)
        zCon = tf.reshape(zP, [-1, 4, 4, 256])
        gen1 = slim.convolution2d_transpose(
            zCon, num_outputs=32, kernel_size=[3, 3], stride=2,
            padding="SAME", normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.relu, scope='g_conv1', weights_initializer=initializer)

        gen2 = slim.convolution2d_transpose(
            gen1, num_outputs=16, kernel_size=[3, 3], stride=2,
            padding="SAME", normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.relu, scope='g_conv2', weights_initializer=initializer)

        gen3 = slim.convolution2d_transpose(
            gen2, num_outputs=8, kernel_size=[3, 3], stride=2,
            padding="SAME", normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.relu, scope='g_conv3', weights_initializer=initializer)
        '''
        gen4 = slim.convolution2d_transpose(
            gen3, num_outputs=3, kernel_size=[3, 3], stride=2,
            padding="SAME", normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.relu, scope='g_conv4', weights_initializer=initializer)
        '''
        g_out = slim.convolution2d_transpose(
            gen3, num_outputs=1, kernel_size=[32, 32], stride=1,
            padding="SAME", biases_initializer=None, activation_fn=tf.nn.tanh, scope='g_out', weights_initializer=initializer)

        g_out_concat = tf.concat(3, [g_out, g_out, g_out], name='concat')

        return g_out_concat


'''
def discriminator(bottom, cat_list, conts, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):

        initializer = tf.truncated_normal_initializer(stddev=0.02)

        dis1 = slim.convolution2d(bottom, 128, [3, 3], padding="SAME", stride=2,
                                  biases_initializer=None, activation_fn=lrelu,
                                  reuse=reuse, scope='d_conv1', weights_initializer=initializer)

        dis2 = slim.convolution2d(dis1, 256, [3, 3], padding="SAME", stride=2,
                                  normalizer_fn=slim.batch_norm, activation_fn=lrelu,
                                  reuse=reuse, scope='d_conv2', weights_initializer=initializer)

        dis3 = slim.convolution2d(dis2, 512, [3, 3], padding="SAME", stride=2,
                                  normalizer_fn=slim.batch_norm, activation_fn=lrelu,
                                  reuse=reuse, scope='d_conv3', weights_initializer=initializer)

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
'''


def discriminator(bottom, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        initializer = tf.truncated_normal_initializer(stddev=0.02)

        x_net, _ = vgg.net('imagenet-vgg-verydeep-19.mat', bottom)

        dis1 = slim.convolution2d(x_net['relu1_2'], 32, [3, 3], padding="SAME",
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
        q_outA = slim.fully_connected(q_a, 10, activation_fn=tf.nn.softmax, reuse=reuse, scope='q_out_cat', weights_initializer=initializer)
        q_cat_outs.append(q_outA)

        q_cont_outs = slim.fully_connected(q_a, 2, activation_fn=tf.nn.tanh, reuse=reuse, scope='q_out_cont', weights_initializer=initializer)

        return d_out, q_cat_outs, q_cont_outs


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def inverse_transform(images):
    return (images + 1.) / 2.


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def data_iterator(images, filenames, batch_size):
    """ A simple data iterator """
    batch_idx = 0
    while True:
        idxs = np.arange(0, len(images))
        np.random.shuffle(idxs)
        for batch_idx in range(0, len(images), batch_size):
            cur_idxs = idxs[batch_idx:batch_idx + batch_size]
            images_batch = images[cur_idxs]
            # images_batch = images_batch.astype("float32")
            names_batch = filenames[cur_idxs]
            yield images_batch, names_batch


def get_dataset(path, dim, channel=3):
    filenames = [join(path, f) for f in listdir(path) if isfile(
        join(path, f)) & f.lower().endswith('bmp')]
    images = np.zeros((len(filenames), dim * dim * channel), dtype=np.uint8)
    # make a dataset
    for i in tqdm.tqdm(range(len(filenames))):
        # for i in tqdm.tqdm(range(10)):
        image = get_image(filenames[i], dim, dim)
        images[i] = image.flatten()
        # get the metadata
    with h5py.File(''.join(['datasets/dataset-rgb-32.h5']), 'w') as f:
        images = f.create_dataset("images", data=images)
        filenames = f.create_dataset('filenames', data=filenames)
    print("dataset loaded")


def get_image(image_path, width, height, mode='RGB'):
    return scipy.misc.imresize(scipy.misc.imread(image_path, mode=mode), [height, width]).astype(np.float)


def discriminator2(x, cat_list, conts, reuse=False):
    # initializer = tf.truncated_normal_initializer(stddev=0.02)
    with tf.variable_scope('discriminator', reuse=reuse):

        x_net, _ = vgg.net('imagenet-vgg-verydeep-19.mat', x)

        dis = slim.fully_connected(slim.flatten(
            x_net['relu4_2']), 1024, normalizer_fn=slim.batch_norm, activation_fn=lrelu, reuse=reuse, scope='d_fc1')

        d_out = slim.fully_connected(
            dis, 1, normalizer_fn=slim.batch_norm, activation_fn=tf.nn.sigmoid, reuse=reuse, scope='d_out')

        q_a = slim.fully_connected(dis, 128, normalizer_fn=slim.batch_norm,
                                   activation_fn=tf.nn.relu, reuse=reuse, scope='q_fc1')

        # Here we define the unique layers used for the q-network. The number of outputs depends on the number of
        # latent variables we choose to define.
        q_cat_outs = []
        for idx, var in enumerate(cat_list):
            q_outA = slim.fully_connected(q_a, var, activation_fn=tf.nn.softmax, reuse=reuse, scope='q_out_cat_' + str(idx))
            q_cat_outs.append(q_outA)

        q_cont_outs = None
        if conts > 0:
            q_cont_outs = slim.fully_connected(q_a, conts, activation_fn=tf.nn.tanh, reuse=reuse, scope='q_out_cont_' + str(conts))

        return d_out, q_cat_outs, q_cont_outs


def gram(layer):
    shape = tf.shape(layer)
    num_images = shape[0]
    num_filters = shape[3]
    size = tf.size(layer)
    filters = tf.reshape(layer, tf.pack([num_images, -1, num_filters]))
    grams = tf.batch_matmul(filters, filters, adj_x=True) / \
        tf.to_float(size)  # / FLAGS.BATCH_SIZE)

    return grams


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


if __name__ == '__main__':
    get_dataset('./train_images/', 32)
