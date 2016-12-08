import numpy as np
import tensorflow as tf
import tqdm  # making loops prettier
import h5py  # for reading our dataset
from os import listdir
from os.path import isfile, join
import scipy.misc
import tensorflow.contrib.slim as slim
import vgg


def gram_np(layer):
    shape = layer.shape
    num_images = shape[0]
    num_filters = shape[3]
    size = layer.size / num_images
    filters = np.reshape(layer, [num_images, -1, num_filters])
    grams = np.matmul(np.swapaxes(filters, 1, 2), filters) / size
    gram_ind = np.triu_indices_from(grams[0])
    return grams[0][gram_ind]


def gram(layer):
    shape = layer.get_shape().as_list()
    num_images = shape[0]
    num_filters = shape[3]
    size = tf.size(layer) / num_images
    filters = tf.reshape(layer, tf.pack([num_images, -1, num_filters]))
    grams = tf.batch_matmul(filters, filters, adj_x=True) / tf.to_double(size)
    triu = np.triu_indices_from(np.zeros((num_filters, num_filters)))
    ind = triu[0] * num_filters + triu[1]
    grams_fla = tf.contrib.layers.flatten(grams)
    return tf.gather(grams_fla[0], ind)


def gram_encoder(gram, reuse=False):
    with tf.variable_scope('gram_enc', reuse=reuse):
        initializer = tf.truncated_normal_initializer(stddev=0.02)
        zGh = slim.fully_connected(gram, 1024, activation_fn=tf.nn.relu, reuse=reuse, scope='gram_project', weights_initializer=initializer)
        zGram = slim.fully_connected(zGh, 32, activation_fn=None, reuse=reuse, scope='gram_project', weights_initializer=initializer)
        return zGram


def generator(z, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):

        initializer = tf.truncated_normal_initializer(stddev=0.02)

        zP = slim.fully_connected(z, 8 * 8 * 256, normalizer_fn=slim.batch_norm,
                                  activation_fn=tf.nn.relu, scope='g_project', weights_initializer=initializer)
        zCon = tf.reshape(zP, [-1, 8, 8, 256])
        gen1 = slim.convolution2d_transpose(
            zCon, num_outputs=64, kernel_size=[3, 3], stride=2,
            padding="SAME", normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.relu, scope='g_conv1', weights_initializer=initializer)

        gen2 = slim.convolution2d_transpose(
            gen1, num_outputs=32, kernel_size=[3, 3], stride=2,
            padding="SAME", normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.relu, scope='g_conv2', weights_initializer=initializer)

        gen3 = slim.convolution2d_transpose(
            gen2, num_outputs=16, kernel_size=[3, 3], stride=2,
            padding="SAME", normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.relu, scope='g_conv3', weights_initializer=initializer)

        gen4 = slim.convolution2d_transpose(
            gen3, num_outputs=8, kernel_size=[3, 3], stride=2,
            padding="SAME", normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.relu, scope='g_conv4', weights_initializer=initializer)

        g_out = slim.convolution2d_transpose(
            gen4, num_outputs=3, kernel_size=[32, 32], stride=1,
            padding="SAME", biases_initializer=None, activation_fn=tf.nn.tanh, scope='g_out', weights_initializer=initializer)

        # g_out_concat = tf.concat(3, [g_out, g_out, g_out], name='concat')

        return g_out


def discriminator(bottom, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        initializer = tf.truncated_normal_initializer(stddev=0.02)

        x_net, _ = vgg.net('imagenet-vgg-verydeep-19.mat', bottom)

        x_layer = x_net['relu2_2']

        dis1 = slim.convolution2d(x_layer, 128, [3, 3], padding="SAME", stride=2,
                                  biases_initializer=None, activation_fn=lrelu,
                                  reuse=reuse, scope='d_conv1', weights_initializer=initializer)
        # dis1 = tf.space_to_depth(dis1, 2)

        dis2 = slim.convolution2d(dis1, 256, [3, 3], padding="SAME", stride=2,
                                  normalizer_fn=slim.batch_norm, activation_fn=lrelu,
                                  reuse=reuse, scope='d_conv2', weights_initializer=initializer)
        # dis2 = tf.space_to_depth(dis2, 2)

        dis3 = slim.convolution2d(dis2, 512, [3, 3], padding="SAME", stride=2,
                                  normalizer_fn=slim.batch_norm, activation_fn=lrelu,
                                  reuse=reuse, scope='d_conv3', weights_initializer=initializer)
        # dis3 = tf.space_to_depth(dis3, 2)

        dis4 = slim.fully_connected(slim.flatten(dis3), 1024, activation_fn=lrelu,
                                    reuse=reuse, scope='d_fc1', weights_initializer=initializer)

        d_out = slim.fully_connected(dis4, 1, activation_fn=tf.nn.sigmoid,
                                     reuse=reuse, scope='d_out', weights_initializer=initializer)

        gram_out = gram(x_layer)

        return d_out, gram_out


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


def get_image(image_path, width, height, mode='RGB'):
    return scipy.misc.imresize(scipy.misc.imread(image_path, mode=mode), [height, width]).astype(np.float)


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


def get_grams(path):
    filenames = [join(path, f) for f in listdir(path) if isfile(join(path, f)) & f.lower().endswith('jpg')]
    grams = []
    mean_pixel = [123.68, 116.779, 103.939]  # ImageNet average from VGG ..
    tf.InteractiveSession()
    for i in tqdm.tqdm(range(len(filenames))):
        image = scipy.misc.imread(filenames[i], mode='RGB').astype(np.float) - mean_pixel
        image = np.expand_dims(image, 0).astype(np.float32)
        style_net, _ = vgg.net('imagenet-vgg-verydeep-19.mat', image)
        style_layer = style_net['relu2_2']
        gram = gram_np(style_layer.eval())
        grams.append(gram)
    with h5py.File(''.join(['datasets/dataset-grams-relu2_2.h5']), 'w') as f:
        grams = f.create_dataset("images", data=grams)
        filenames = f.create_dataset('filenames', data=filenames)
    print("dataset loaded")


if __name__ == '__main__':
    # get_dataset('./train_images/', 32)
    get_grams('./styles/')
