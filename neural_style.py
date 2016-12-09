import time
import tensorflow as tf
import vgg
import h5py  # for reading our dataset
from os import listdir
from os.path import isfile, join
import numpy as np
import tqdm  # making loops prettier
import scipy.misc
import os
from tensorflow.python.client import device_lib
import neural_model
import model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print device_lib.list_local_devices()

tf.app.flags.DEFINE_integer("CONTENT_WEIGHT", 5, "5e0Weight for content features loss")
tf.app.flags.DEFINE_integer("STYLE_WEIGHT", 30, "1e2Weight for style features loss")
tf.app.flags.DEFINE_integer("TV_WEIGHT", 1e-5, "Weight for total variation loss")
tf.app.flags.DEFINE_string("VGG_PATH", "imagenet-vgg-verydeep-19.mat", "Path to vgg model weights")
tf.app.flags.DEFINE_string("CONTENT_LAYERS", "relu3_4", "Which VGG layer to extract content loss from")
tf.app.flags.DEFINE_string("STYLE_LAYERS", "relu1_2,relu2_2,relu3_4,relu4_4", "Which layers to extract style from")
# tf.app.flags.DEFINE_string("STYLE_LAYERS", "relu3_4", "Which layers to extract style from")
tf.app.flags.DEFINE_float("LEARNING_RATE", 1e-3, "Learning rate")
tf.app.flags.DEFINE_string("MODEL_DIR", "style_model", "path")

FLAGS = tf.app.flags.FLAGS

mean_pixel = [123.68, 116.779, 103.939]  # ImageNet average from VGG ..


def data_iterator(images, filenames, batch_size):
    """ A simple data iterator """
    batch_idx = 0
    while True:
        idxs = np.arange(0, len(images))
        np.random.shuffle(idxs)
        for batch_idx in range(0, len(images), batch_size):
            cur_idxs = idxs[batch_idx:batch_idx + batch_size]
            images_batch = images[cur_idxs]
            names_batch = filenames[cur_idxs]
            yield images_batch, names_batch


def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0, 0, 0, 0], tf.pack([-1, height - 1, -1, -1])) - tf.slice(layer, [0, 1, 0, 0], [-1, -1, -1, -1])
    x = tf.slice(layer, [0, 0, 0, 0], tf.pack([-1, -1, width - 1, -1])) - tf.slice(layer, [0, 0, 1, 0], [-1, -1, -1, -1])
    return tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))

# TODO: Okay to flatten all style images into one gram?


def gram(layer):
    shape = layer.get_shape().as_list()
    num_images = shape[0]
    num_filters = shape[3]
    size = tf.size(layer) / num_images
    filters = tf.reshape(layer, tf.pack([num_images, -1, num_filters]))
    grams = tf.batch_matmul(filters, filters, adj_x=True) / tf.to_float(size)
    return grams


def get_style_features(style_image, style_layers):
    net, _ = vgg.net(FLAGS.VGG_PATH, style_image)
    features = []
    for layer in style_layers:
        features.append(gram(net[layer]))
    return features


def get_content_features(content_image, content_layers):
    net, _ = vgg.net(FLAGS.VGG_PATH, content_image)
    layers = []
    for layer in content_layers:
        layers.append(net[layer])
    return layers


def fast_style():
    batch_size = 4
    num_epochs = 1
    style_layers = FLAGS.STYLE_LAYERS.split(',')
    content_layers = FLAGS.CONTENT_LAYERS.split(',')
    with h5py.File(''.join(['datasets/coco-256.h5']), 'r') as hf:
        content_images = hf['images'].value
        content_names = hf['filenames'].value

    total_batch = int(np.floor(len(content_images) / (batch_size)))

    total_loss = 0
    total_content = 0
    total_style = 0
    total_variation = 0

    style_names = [join('styles', f) for f in listdir('styles') if isfile(join('styles', f)) & f.lower().endswith('jpg')]

    style_holder = tf.placeholder(shape=[1, None, None, 3], dtype=tf.float32)  # Real images
    content_holder = tf.placeholder(shape=[batch_size, 256, 256, 3], dtype=tf.float32)  # Random vector

    # generated = neural_model.net(content_holder)
    generated = [model.net(content_holder)]

    style_net, _ = vgg.net(FLAGS.VGG_PATH, style_holder)
    content_net, _ = vgg.net(FLAGS.VGG_PATH, content_holder)

    for i in range(len(generated)):
        generated_net, _ = vgg.net(FLAGS.VGG_PATH, generated[i])

        content_loss = 0
        for layer in content_layers:
            content_vgg = content_net[layer]
            generated_vgg = generated_net[layer]
            size = tf.size(generated_vgg)
            content_loss += tf.nn.l2_loss(generated_vgg - content_vgg) / tf.to_float(size)
        content_loss = content_loss / len(content_layers)

        style_loss = 0
        for layer in style_layers:
            generated_vgg = generated_net[layer]
            style_vgg = style_net[layer]
            size = tf.square(style_vgg.get_shape().as_list()[3]) * batch_size
            # for style_batch in style_gram:
            style_loss += tf.nn.l2_loss(gram(generated_vgg) - gram(style_vgg)) / tf.to_float(size)
        style_loss = style_loss / len(style_layers)

        total_content += content_loss
        total_style += style_loss
        total_variation += total_variation_loss(generated[i])

    total_loss = FLAGS.STYLE_WEIGHT * total_style + FLAGS.CONTENT_WEIGHT * total_content + FLAGS.TV_WEIGHT * total_variation
    output_format = tf.saturate_cast(tf.concat(0, [generated[-1], content_holder]) + mean_pixel, tf.uint8)

    tvars = tf.trainable_variables()
    train_op = tf.train.AdamOptimizer(FLAGS.LEARNING_RATE)
    grads = train_op.compute_gradients(total_loss, tvars)
    update = train_op.apply_gradients(grads)
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=0)
    sess = tf.Session()

    for i in xrange(len(style_names)):

        sess.run(tf.initialize_all_variables())
        sess.run(tf.initialize_local_variables())

        print 'style: ' + style_names[i]
        style_image = scipy.misc.imread(style_names[i], mode='RGB') - mean_pixel
        style_image = np.expand_dims(style_image, 0)
        epoch = 0

        while epoch < num_epochs:
            content_iter = data_iterator(content_images, content_names, batch_size)
            for j in tqdm.tqdm(xrange(total_batch)):
                content_image, content_name = content_iter.next()
                content_image = np.reshape(content_image, [batch_size, 256, 256, 3]) - mean_pixel
                _, loss_t, loss_s, loss_c = sess.run([update, total_loss, total_style, total_content], feed_dict={content_holder: content_image, style_holder: style_image})

                if j % 100 == 0:
                    print 'epoch: ' + str(epoch) + ' loss: ' + str(loss_t) + ' loss_s: ' + str(loss_s) + ' loss_c: ' + str(loss_c)
                    output_t = sess.run(output_format, feed_dict={content_holder: content_image})
                    for j, raw_image in enumerate(output_t):
                        scipy.misc.imsave('test/out%s-%s.png' % (epoch, j + 1), raw_image)
            if epoch % 1 == 0:
                if not os.path.exists(FLAGS.MODEL_DIR):
                    os.makedirs(FLAGS.MODEL_DIR)
                saver.save(sess, FLAGS.MODEL_DIR + '/model-' + str(style_names[i][7:-4]) + '.ckpt', global_step=i)
                print "Saved Model"
            epoch += 1


if __name__ == '__main__':
    fast_style()
