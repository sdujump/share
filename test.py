from scipy import misc
import os
import tensorflow as tf
from os import listdir
from os.path import isfile, join
from tensorflow.python.client import device_lib
import numpy as np
from skimage import io, exposure

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
print device_lib.list_local_devices()


tf.app.flags.DEFINE_string(
    "TRAIN_IMAGES_PATH", "test_images", "Path to training images")


FLAGS = tf.app.flags.FLAGS


def get_image(path, epochs, shuffle=False, crop=True):
    filenames = [join(path, f) for f in listdir(path) if isfile(
        join(path, f)) & f.lower().endswith('bmp')]
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=epochs, shuffle=shuffle)
    reader = tf.WholeFileReader()
    # reader = tf.FixedLengthRecordReader(record_bytes=1862)
    img_key, img_bytes = reader.read(filename_queue)
    # image = tf.image.decode_jpeg(img_bytes, channels=3)
    image = tf.decode_raw(img_bytes, tf.uint8)
    slice_image = tf.image.rot90(tf.expand_dims(tf.transpose(tf.reshape(
        tf.gather(image, tf.range(1862 - 784, 1862)), (28, 28))), 2), 1)
    return slice_image, img_key


def show_variables(variales):
    for i in range(len(variales)):
        print(variales[i].name)


def main(argv=None):

    content_paths = FLAGS.TRAIN_IMAGES_PATH
    '''
    beta = tf.get_variable('beta', initializer=tf.zeros_initializer([10]))
    alpha = tf.Variable(tf.zeros([10]), name='alpha')

    with tf.Session() as tess:
        tess.run(tf.initialize_all_variables())
        bbb, aaa = tess.run([beta, alpha])
        print bbb
        print aaa
    '''
    input_image, img_key = get_image(content_paths, 1, False)
    input_images, img_keys = tf.train.batch((input_image, img_key), 2, dynamic_pad=True)

    with tf.Session() as sess:

        sess.run(tf.initialize_all_variables())
        sess.run(tf.initialize_local_variables())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop():
                output_t, output_n = sess.run([input_images, img_keys])
                for i, raw_image in enumerate(output_t):
                    filepath = output_n[i]
                    filename = filepath[filepath.find("/") + 1:filepath.find(".")]
                    io.imsave('test/%s_%s.png' % ('rec', filename), np.reshape(raw_image.astype("uint8"), (28, 28)))
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
