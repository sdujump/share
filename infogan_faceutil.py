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


def generator(z, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):

        initializer = tf.truncated_normal_initializer(stddev=0.02)

        zP = slim.fully_connected(z, 4 * 4 * 512, normalizer_fn=slim.batch_norm,
                                  activation_fn=tf.nn.relu, scope='g_project', weights_initializer=initializer)
        zCon = tf.reshape(zP, [-1, 4, 4, 512])
        gen1 = slim.convolution2d_transpose(
            zCon, num_outputs=256, kernel_size=[5, 5], stride=2,
            padding="SAME", normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.relu, scope='g_conv1', weights_initializer=initializer)

        gen2 = slim.convolution2d_transpose(
            gen1, num_outputs=128, kernel_size=[5, 5], stride=2,
            padding="SAME", normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.relu, scope='g_conv2', weights_initializer=initializer)

        gen3 = slim.convolution2d_transpose(
            gen2, num_outputs=64, kernel_size=[5, 5], stride=2,
            padding="SAME", normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.relu, scope='g_conv3', weights_initializer=initializer)
        '''
        gen4 = slim.convolution2d_transpose(
            gen3, num_outputs=3, kernel_size=[3, 3], stride=2,
            padding="SAME", normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.relu, scope='g_conv4', weights_initializer=initializer)
        '''
        g_out = slim.convolution2d_transpose(
            gen3, num_outputs=3, kernel_size=[5, 5], stride=2,
            padding="SAME", activation_fn=tf.nn.tanh, scope='g_out', weights_initializer=initializer)

        # g_out_concat = tf.concat(3, [g_out, g_out, g_out], name='concat')

        return g_out


def discriminator(bottom, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        initializer = tf.truncated_normal_initializer(stddev=0.02)

        dis1 = slim.convolution2d(bottom, 64, [5, 5], padding="SAME", stride=2, activation_fn=lrelu,
                                  reuse=reuse, scope='d_conv1', weights_initializer=initializer)
        # dis1 = tf.space_to_depth(dis1, 2)

        dis2 = slim.convolution2d(dis1, 128, [5, 5], padding="SAME", stride=2,
                                  normalizer_fn=slim.batch_norm, activation_fn=lrelu,
                                  reuse=reuse, scope='d_conv2', weights_initializer=initializer)
        # dis2 = tf.space_to_depth(dis2, 2)

        dis3 = slim.convolution2d(dis2, 256, [5, 5], padding="SAME", stride=2,
                                  normalizer_fn=slim.batch_norm, activation_fn=lrelu,
                                  reuse=reuse, scope='d_conv3', weights_initializer=initializer)
        # dis3 = tf.space_to_depth(dis3, 2)

        dis4 = slim.convolution2d(dis3, 512, [5, 5], padding="SAME", stride=2,
                                  normalizer_fn=slim.batch_norm, activation_fn=lrelu,
                                  reuse=reuse, scope='d_conv4', weights_initializer=initializer)
        '''
        dis4 = slim.fully_connected(slim.flatten(dis4), 1024, activation_fn=lrelu,
                                    reuse=reuse, scope='d_fc1', weights_initializer=initializer)
        '''
        d_out = slim.fully_connected(slim.flatten(dis4), 1, activation_fn=tf.nn.sigmoid,
                                     reuse=reuse, scope='d_out', weights_initializer=initializer)

        q_a = slim.fully_connected(slim.flatten(dis4), 128, normalizer_fn=slim.batch_norm,
                                   reuse=reuse, scope='q_a', weights_initializer=initializer)

        # Here we define the unique layers used for the q-network. The number of outputs depends on the number of
        # latent variables we choose to define.

        q_cont_outs1 = slim.fully_connected(q_a, 2, activation_fn=tf.nn.tanh, reuse=reuse, scope='q_out_cont1', weights_initializer=initializer)
        # q_cont_outs2 = slim.fully_connected(q_a, 2, activation_fn=tf.nn.tanh, reuse=reuse, scope='q_out_cont2', weights_initializer=initializer)

        return d_out, q_cont_outs1


'''
def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)
'''


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


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
        # np.random.shuffle(idxs)
        for batch_idx in range(0, len(images), batch_size):
            cur_idxs = idxs[batch_idx:batch_idx + batch_size]
            cur_idxs = np.sort(cur_idxs)
            images_batch = images[cur_idxs.tolist()]
            # images_batch = images_batch.astype("float32")
            names_batch = filenames[cur_idxs.tolist()]
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


def gram(layer):
    shape = tf.shape(layer)
    num_images = shape[0]
    num_filters = shape[3]
    size = tf.size(layer)
    filters = tf.reshape(layer, tf.pack([num_images, -1, num_filters]))
    grams = tf.batch_matmul(filters, filters, adj_x=True) / \
        tf.to_float(size)  # / FLAGS.BATCH_SIZE)

    return grams


if __name__ == '__main__':
    get_dataset('./train_images/', 32)
