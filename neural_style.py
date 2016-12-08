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
import model2 as model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print device_lib.list_local_devices()

tf.app.flags.DEFINE_integer("CONTENT_WEIGHT", 5, "5e0Weight for content features loss")
tf.app.flags.DEFINE_integer("STYLE_WEIGHT", 30, "1e2Weight for style features loss")
tf.app.flags.DEFINE_integer("TV_WEIGHT", 1e-5, "Weight for total variation loss")
tf.app.flags.DEFINE_string("VGG_PATH", "imagenet-vgg-verydeep-19.mat", "Path to vgg model weights")
tf.app.flags.DEFINE_string("CONTENT_LAYERS", "relu3_4", "Which VGG layer to extract content loss from")
tf.app.flags.DEFINE_string("STYLE_LAYERS", "relu1_2,relu2_2,relu3_4,relu4_4", "Which layers to extract style from")
tf.app.flags.DEFINE_float("STYLE_SCALE", 1.0, "Scale styles. Higher extracts smaller features")
tf.app.flags.DEFINE_float("LEARNING_RATE", 10., "Learning rate")
tf.app.flags.DEFINE_integer("NUM_ITERATIONS", 300, "Number of iterations")
tf.app.flags.DEFINE_string("MODEL_DIR", "style_model", "path")

FLAGS = tf.app.flags.FLAGS

mean_pixel = [123.68, 116.779, 103.939]  # ImageNet average from VGG ..


def get_image(image_path, width, height, mode='RGB'):
    return scipy.misc.imresize(scipy.misc.imread(image_path, mode=mode), [height, width]).astype(np.float)


def get_dataset(path, dim, channel=3):
    filenames = [join(path, f) for f in listdir(path) if isfile(
        join(path, f)) & f.lower().endswith('jpg')]
    images = np.zeros((len(filenames), dim * dim * channel), dtype=np.uint8)
    # make a dataset
    for i in tqdm.tqdm(range(len(filenames))):
        # for i in tqdm.tqdm(range(10)):
        image = get_image(filenames[i], dim, dim)
        images[i] = image.flatten()
        # get the metadata
    with h5py.File(''.join(['datasets/coco-256.h5']), 'w') as f:
        images = f.create_dataset("images", data=images)
        filenames = f.create_dataset('filenames', data=filenames)
    print("dataset loaded")


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

    '''
def gram(layer):
    shape = layer.get_shape().as_list()
    num_filters = shape[3]
    size = tf.size(layer)
    filters = tf.reshape(layer, tf.pack([-1, num_filters]))
    gram = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(size)

    return gram
'''
# TODO: Different style scales per image.


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


def inference(path, name):
    with h5py.File(''.join(['datasets/coco-256.h5']), 'r') as hf:
        content_images = hf['images'].value
        content_names = hf['filenames'].value
    content_holder = tf.placeholder(shape=[1, 256, 256, 3], dtype=tf.float32)  # Random vector
    total_batch = len(content_images)
    content_iter = data_iterator(content_images, content_names, 1)

    generated = model.net(content_holder)
    output_format = tf.saturate_cast(generated + mean_pixel, tf.uint8)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())
    saver = tf.train.Saver()
    saver.restore(sess, path)

    for j in tqdm.tqdm(xrange(total_batch)):
        content_image, content_name = content_iter.next()
        print "stylize: " + str(content_name)
        content_image = np.reshape(content_image, [1, 256, 256, 3]) - mean_pixel
        output_t = sess.run(output_format, feed_dict={content_holder: content_image})
        scipy.misc.imsave('coco_style/%s-%s.png' % (content_name[0][5:-4], name), output_t[0])


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
    generated = [model.net(content_holder, training=True)]

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
            # size = tf.size(style_vgg)
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
    train_op = tf.train.AdamOptimizer(1e-3)
    grads = train_op.compute_gradients(total_loss, tvars)
    update = train_op.apply_gradients(grads)
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=0)
    sess = tf.Session()

    for i in xrange(len(style_names)):

        sess.run(tf.initialize_all_variables())
        sess.run(tf.initialize_local_variables())

        print 'style: ' + style_names[i]
        # style_image = (scipy.misc.imread(style_names[i], mode='RGB') / 255.0 - 0.5) * 2.0
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


def stylize():
    style_layers = FLAGS.STYLE_LAYERS.split(',')
    content_layers = FLAGS.CONTENT_LAYERS.split(',')
    with h5py.File(''.join(['datasets/coco-256.h5']), 'r') as hf:
        content_images = hf['images'].value
        content_names = hf['filenames'].value

    style_names = [join('styles', f) for f in listdir('styles') if isfile(join('styles', f)) & f.lower().endswith('jpg')]

    style_holder = tf.placeholder(shape=[1, None, None, 3], dtype=tf.float32)  # Real images
    content_holder = tf.placeholder(shape=[1, 256, 256, 3], dtype=tf.float32)  # Random vector
    # random = tf.random_normal(content_holder.get_shape().as_list())
    # random = tf.Variable(tf.random_normal(content_holder.get_shape().as_list(), 0, 1, dtype=tf.float32), name='random', trainable=False)
    # opt_image = tf.Variable(0.4 * random + 0.6 * content_holder, name='opt')
    opt_image = tf.Variable(content_holder, name='opt')

    style_features_t = get_style_features(style_holder, style_layers)
    content_features_t = get_content_features(content_holder, content_layers)
    opt_image_r, _ = vgg.net(FLAGS.VGG_PATH, opt_image)

    content_loss = 0
    for content_features, layer in zip(content_features_t, content_layers):
        layer_size = tf.size(content_features)
        content_loss += tf.nn.l2_loss(opt_image_r[layer] - content_features) / tf.to_float(layer_size)
    content_loss = FLAGS.CONTENT_WEIGHT * content_loss / len(content_layers)

    style_loss = 0
    for style_gram, layer in zip(style_features_t, style_layers):
        layer_size = tf.size(style_gram)
        style_loss += tf.nn.l2_loss(gram(opt_image_r[layer]) - style_gram) / tf.to_float(layer_size)
    style_loss = FLAGS.STYLE_WEIGHT * style_loss / len(style_layers)

    tv_loss = FLAGS.TV_WEIGHT * total_variation_loss(opt_image)

    total_loss = content_loss + style_loss + tv_loss
    tvars = tf.trainable_variables()
    train_op = tf.train.AdamOptimizer(FLAGS.LEARNING_RATE)
    grads = train_op.compute_gradients(total_loss, tvars)
    update = train_op.apply_gradients(grads)
    # saver = tf.train.Saver(tf.all_variables())
    sess = tf.Session()
    for i in xrange(len(style_names)):
        # style_image = (scipy.misc.imread(style_names[i], mode='RGB') / 255.0 - 0.5) * 2.0
        style_image = scipy.misc.imread(style_names[i], mode='RGB') - mean_pixel
        style_image = np.expand_dims(style_image, 0)
        content_iter = data_iterator(content_images, content_names, 1)
        for j in xrange(len(content_images)):
            content_image, content_name = content_iter.next()
            print 'style: ' + style_names[i] + ' content: ' + content_name[0]
            content_image = np.reshape(content_image, [1, 256, 256, 3]) - mean_pixel
            # content_image = (np.reshape(content_image, [1, 256, 256, 3]) / 255.0 - 0.5) * 2.0
            # image_t = content_image
            init = tf.initialize_all_variables()
            sess.run(init, feed_dict={content_holder: content_image})
            for step in tqdm.tqdm(xrange(FLAGS.NUM_ITERATIONS)):
                _, loss_t, image_t = sess.run([update, total_loss, opt_image], feed_dict={content_holder: content_image, style_holder: style_image})
                if step % 10 == 0:
                    print 'step: ' + str(step) + ' loss: ' + str(loss_t)
            # image_t = sess.run(opt_image)
            # scipy.misc.imsave('coco_style/' + content_name[0][5:-4] + '-%s.jpg' % (i), (np.squeeze(image_t) + 1) / 2)
            scipy.misc.imsave('coco_style/' + content_name[0][5:-4] + '-%s.jpg' % (i), scipy.misc.bytescale(np.squeeze(image_t) + mean_pixel))
            scipy.misc.imsave('coco_style/' + content_name[0][5:-4] + '.jpg', np.squeeze(content_image) + mean_pixel)


def main(argv=None):
    style_paths = FLAGS.STYLE_IMAGES
    style_layers = FLAGS.STYLE_LAYERS.split(',')
    content_path = FLAGS.CONTENT_IMAGE
    content_layers = FLAGS.CONTENT_LAYERS.split(',')

    style_features_t = get_style_features(style_paths, style_layers)
    content_features_t, image_t = get_content_features(content_path, content_layers)

    image = tf.constant(image_t)
    random = tf.random_normal(image_t.shape)
    initial = tf.Variable(random if FLAGS.RANDOM_INIT else image)

    net, _ = vgg.net(FLAGS.VGG_PATH, initial)

    content_loss = 0
    for content_features, layer in zip(content_features_t, content_layers):
        layer_size = tf.size(content_features)
        content_loss += tf.nn.l2_loss(net[layer] -
                                      content_features) / tf.to_float(layer_size)
    content_loss = FLAGS.CONTENT_WEIGHT * content_loss / len(content_layers)

    style_loss = 0
    for style_gram, layer in zip(style_features_t, style_layers):
        layer_size = tf.size(style_gram)
        style_loss += tf.nn.l2_loss(gram(net[layer]) -
                                    style_gram) / tf.to_float(layer_size)
    style_loss = FLAGS.STYLE_WEIGHT * style_loss / (len(style_layers) * len(style_paths))

    tv_loss = FLAGS.TV_WEIGHT * total_variation_loss(initial)

    total_loss = content_loss + style_loss + tv_loss

    train_op = tf.train.AdamOptimizer(FLAGS.LEARNING_RATE).minimize(total_loss)

    output_image = tf.image.encode_png(tf.saturate_cast(tf.squeeze(initial) + mean_pixel, tf.uint8))

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        start_time = time.time()
        for step in range(FLAGS.NUM_ITERATIONS):
            _, loss_t = sess.run([train_op, total_loss])
            elapsed = time.time() - start_time
            start_time = time.time()
            print(step, elapsed, loss_t)
        image_t = sess.run(output_image)
        with open('out.png', 'wb') as f:
            f.write(image_t)


if __name__ == '__main__':
    # tf.app.run()
    # get_dataset('coco', 256, channel=3)
    fast_style()
    # inference('style_model/model-starry_night.ckpt-3', '0')
