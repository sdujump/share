# -*- coding: utf-8 -*-
import os
from os import listdir
from os.path import isfile, join
import sys
import time
import re
import numpy as np
import tensorflow as tf
import vaemodel
from skimage import io, exposure

tf.app.flags.DEFINE_string("train_image_dir", "train_images", "")
tf.app.flags.DEFINE_string("model_dir", "model", "")
tf.app.flags.DEFINE_integer("max_epoch", 1000, "")
tf.app.flags.DEFINE_integer("num_trains_per_epoch", 1, "")
tf.app.flags.DEFINE_integer("batchsize", 1, "")
tf.app.flags.DEFINE_integer("n_steps_to_optimize_dis", 1, "")
tf.app.flags.DEFINE_integer("ndim_y", 10, "")
tf.app.flags.DEFINE_integer("ndim_x", 28 * 28, "")
tf.app.flags.DEFINE_integer("ndim_z", 2, "")
tf.app.flags.DEFINE_integer("gpu_enabled", 0, "")

FLAGS = tf.app.flags.FLAGS


def show_variables(variales):
    for i in range(len(variales)):
        print(variales[i].name)


def get_image(path, epochs, shuffle=False, crop=True):
    filenames = [join(path, f) for f in listdir(path) if isfile(
        join(path, f)) & f.lower().endswith('bmp')]
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=epochs, shuffle=shuffle)
    # reader = tf.WholeFileReader()
    reader = tf.FixedLengthRecordReader(record_bytes=1862)
    img_key, img_bytes = reader.read(filename_queue)
    # image = tf.image.decode_jpeg(img_bytes, channels=3)
    image = tf.decode_raw(img_bytes, tf.uint8)
    slice_image = tf.image.rot90(tf.expand_dims(tf.transpose(tf.reshape(
        tf.gather(image, tf.range(1862 - 784, 1862)), (28, 28))), 2), 1)
    return slice_image, img_key


def batch_data(batchsize, ndim_x, ndim_y, dataset, labels, gpu_enabled=0):
    return batchsize


def load_labeled_images(image_dir):
    dataset = []
    labels = []
    fs = os.listdir(image_dir)
    i = 0
    for fn in fs:
        if fn.endswith(".bmp"):
            m = re.match("([0-9]+)_.+", fn)
            label = int(m.group(1))
            img = io.imread("%s/%s" % (image_dir, fn)) / 255.0
            dataset.append(img)
            labels.append(label)
            i += 1
            if i % 100 == 0:
                sys.stdout.write(
                    "\rloading images...({:d} / {:d})".format(i, len(fs)))
                sys.stdout.flush()
    sys.stdout.write("\n")
    return dataset, labels


def minimize_and_clip(optimizer, objective, var_list, step, clip_val=5):
    gradients = optimizer.compute_gradients(objective, var_list=var_list)
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
    return optimizer.apply_gradients(gradients, global_step=step)


def train_autoencoder():
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)
    image_paths = FLAGS.train_image_dir
    ndim_z = FLAGS.ndim_z
    batchsize = FLAGS.batchsize
    input_image, _ = get_image(image_paths, 1, False)
    input_images = tf.train.batch([input_image], batchsize, dynamic_pad=True)
    size = 784

    reconst, latent = vaemodel.encoder_decoder(input_images, size, ndim_z, size)
    rec_loss = tf.nn.l2_loss(tf.to_float(input_images) - reconst) / tf.to_float(size * batchsize)
    rec_vars = [v for v in tf.all_variables() if v.name.startswith("encoder_decoder/")]
    rec_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    rec_step = tf.Variable(0, name="rec_step", trainable=False)
    rec_op = minimize_and_clip(rec_optimizer, objective=rec_loss, var_list=rec_vars, step=rec_step)

    pz_priors = vaemodel.get_prior([batchsize] + [ndim_z])
    dis_latent_outputs = vaemodel.discriminator(latent)
    dis_prior_outputs = vaemodel.discriminator(pz_priors, reuse=True)

    # discriminator_outputs = tf.concat(0, [dis_latent_outputs, dis_prior_outputs])
    # discriminator_target = tf.concat(0, [tf.zeros([batchsize, 1]), tf.ones([batchsize, 1])])

    dis_loss = tf.reduce_mean(tf.log(dis_prior_outputs) + tf.log(1. - dis_latent_outputs))
    dis_vars = [v for v in tf.all_variables() if v.name.startswith("discriminator/")]
    dis_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    dis_step = tf.Variable(0, name="dis_step", trainable=False)
    dis_op = minimize_and_clip(dis_optimizer, objective=-dis_loss, var_list=dis_vars, step=dis_step)

    gen_loss = tf.reduce_mean(tf.log(dis_latent_outputs))
    gen_vars = [v for v in tf.all_variables() if v.name.startswith("encoder_decoder/encoder/")]
    gen_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    gen_step = tf.Variable(0, name="gen_step", trainable=False)
    gen_op = minimize_and_clip(gen_optimizer, objective=-gen_loss, var_list=gen_vars, step=gen_step)

    reconst_img = tf.reshape(reconst, tf.shape(input_images))

    with tf.Session() as sess:
            saver = tf.train.Saver(tf.all_variables())
            file = tf.train.latest_checkpoint(FLAGS.model_dir)
            if file:
                print('Restoring model from {}'.format(file))
                saver.restore(sess, file)
            else:
                print('New model initilized')
                sess.run(tf.initialize_all_variables())
                sess.run(tf.initialize_local_variables())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            start_time = time.time()
            try:
                while not coord.should_stop():
                    recloss, _, recstep = sess.run([rec_loss, rec_op, rec_step])
                    disloss, _, disstep = sess.run([dis_loss, dis_op, dis_step])
                    genloss, _, genstep = sess.run([gen_loss, gen_op, gen_step])
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    if recstep % 1 == 0:
                        print(recstep, recloss, elapsed_time)
                        output_t = sess.run(reconst_img)
                        for i, raw_image in enumerate(output_t):
                            io.imsave('test/out%s-%s.png' % (recstep, i + 1), raw_image)
                    if recstep % 100 == 0:
                        saver.save(sess, FLAGS.model_dir + 'autoencoder-model', global_step=recstep)
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                coord.request_stop()

            coord.join(threads)


if __name__ == '__main__':
    train_autoencoder()
