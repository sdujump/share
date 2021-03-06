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

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("model", "", "path")
tf.app.flags.DEFINE_string("tag", "", "tag")
tf.app.flags.DEFINE_string("usegpu", "", "gpu")


os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.usegpu
print device_lib.list_local_devices()

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
            # images_batch = images_batch.astype("float32")
            names_batch = filenames[cur_idxs]
            yield images_batch, names_batch


def main(argv=None):
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
    saver.restore(sess, "style_model/" + FLAGS.model)

    for j in tqdm.tqdm(xrange(total_batch)):
        content_image, content_name = content_iter.next()
        print "stylize: " + str(content_name)
        content_image = np.reshape(content_image, [1, 256, 256, 3]) - mean_pixel
        output_t = sess.run(output_format, feed_dict={content_holder: content_image})
        scipy.misc.imsave('coco_style/%s-%s.png' % (content_name[0][5:-4], FLAGS.tag), output_t[0])


if __name__ == '__main__':
    tf.app.run()
    # get_dataset('coco', 256, channel=3)
    # fast_style()
    # inference('style_model/' + FLAGS.model, )
