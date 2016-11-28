# -*- coding: utf-8 -*-
import os
from os import listdir
from os.path import isfile, join
import sys
import time
import re
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import cvaemodel
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# matplotlib.use('Agg')

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
print device_lib.list_local_devices()

tf.app.flags.DEFINE_string("train_image_dir", "train_images", "")
tf.app.flags.DEFINE_string("test_image_dir", "test_images", "")
tf.app.flags.DEFINE_string("model_dir", "model2", "")
tf.app.flags.DEFINE_integer("max_epoch", 1000, "")
tf.app.flags.DEFINE_integer("num_trains_per_epoch", 1, "")
tf.app.flags.DEFINE_integer("batchsize", 100, "")
tf.app.flags.DEFINE_integer("n_steps_to_optimize_dis", 1, "")
tf.app.flags.DEFINE_integer("ndim_y", 10, "")
tf.app.flags.DEFINE_integer("ndim_x", 28 * 28, "")
tf.app.flags.DEFINE_integer("ndim_z", 20, "")
tf.app.flags.DEFINE_integer("gpu_enabled", 0, "")

FLAGS = tf.app.flags.FLAGS


def show_variables(variales):
    for i in range(len(variales)):
        print(variales[i].name)


def get_image(path, epochs, shuffle=False):
    filenames = [join(path, f) for f in listdir(path) if isfile(
        join(path, f)) & f.lower().endswith('bmp')]
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=epochs, shuffle=shuffle)
    reader = tf.WholeFileReader()
    # reader = tf.FixedLengthRecordReader(record_bytes=1862)
    img_key, img_bytes = reader.read(filename_queue)
    # image = tf.image.decode_jpeg(img_bytes, channels=3)
    image = tf.to_float(tf.decode_raw(img_bytes, tf.uint8)) / 255.0
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


def train_gan():
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)
    image_paths = FLAGS.train_image_dir
    ndim_z = FLAGS.ndim_z
    batchsize = FLAGS.batchsize
    input_image, _ = get_image(image_paths, 75, True)
    input_image = tf.reshape(input_image, [28, 28, 1])
    input_images = tf.train.batch([input_image], batchsize, dynamic_pad=True)
    # input_images = tf.reshape(input_images)
    # input_images, img_keys = tf.train.batch((input_image, img_key), batchsize, dynamic_pad=True)

    # all_input = tf.placeholder(tf.float32, [batchsize, 28, 28, 1])

    # reconst, z_mean, z_log_sigma_sq, z = cvaemodel.conv_encoder_z_decoder(input_images, ndim_z)
    reconst, z_mean, z_log_sigma_sq, dis_generate, dis_input, dis_p = cvaemodel.inference(input_images, ndim_z)
    reconst_img = reconst * 255.0

    # flat_input = tf.to_float(tf.reshape(input_images, [batchsize, -1]))
    # flat_recon = tf.to_float(tf.reshape(reconst, [batchsize, -1]))
    # rec_loss = -tf.reduce_sum(flat_input * tf.log(1e-3 + flat_recon) + (1. - flat_input) * tf.log(1e-3 + 1. - flat_recon), 1)
    rec_loss = tf.reduce_sum(tf.square(input_images - reconst), [1, 2, 3])  # / tf.to_float(784)
    latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1)
    total_loss = tf.reduce_mean(rec_loss + latent_loss)
    rec_vars = [v for v in tf.all_variables() if v.name.startswith("inference/encoder/") or v.name.startswith("inference/decoder/")]
    rec_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    rec_step = tf.Variable(0, name="rec_step", trainable=False)
    rec_op = minimize_and_clip(rec_optimizer, objective=total_loss, var_list=rec_vars, step=rec_step)

    gen_loss = -tf.reduce_sum(tf.log(dis_generate))
    gen_vars = [v for v in tf.all_variables() if v.name.startswith("inference/decoder/") or v.name.startswith("inference/decoder/")]
    gen_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    gen_step = tf.Variable(0, name="gen_step", trainable=False)
    gen_op = minimize_and_clip(gen_optimizer, objective=gen_loss, var_list=gen_vars, step=gen_step)

    # dis_generate = cvaemodel.conv_discriminator(reconst)
    # dis_input = cvaemodel.conv_discriminator(input_images, reuse=True)

    dis_loss = - (tf.reduce_sum(tf.log(dis_input) + tf.log(1. - dis_generate) + tf.log(1. - dis_p)))
    dis_vars = [v for v in tf.all_variables() if v.name.startswith("inference/discriminator/")]
    dis_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    dis_step = tf.Variable(0, name="dis_step", trainable=False)
    dis_op = minimize_and_clip(dis_optimizer, objective=dis_loss, var_list=dis_vars, step=dis_step)

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
            try:
                for step in xrange(1000000):
                    if coord.should_stop():
                        break
                    start_time = time.time()
                    enloss, _, = sess.run([total_loss, rec_op])
                    enloss_elapsed_time = time.time() - start_time

                    if step > 400:
                        start_time = time.time()
                        disloss, _, = sess.run([dis_loss, dis_op])
                        disloss_elapsed_time = time.time() - start_time
                    if step > 800:
                        start_time = time.time()
                        genloss, _, = sess.run([gen_loss, gen_op])
                        genloss_elapsed_time = time.time() - start_time
                    if step % 100 == 0:
                        if step > 400:
                            print('discriminator', step, disloss, disloss_elapsed_time)
                        if step > 800:
                            print('generator', step, genloss, genloss_elapsed_time)
                        print('reconstruct', step, enloss, enloss_elapsed_time)
                        output_t = sess.run(reconst_img)
                        for i, raw_image in enumerate(output_t):
                            io.imsave('test/out%s-%s.png' % (step, i + 1), np.reshape(raw_image.astype("uint8"), (28, 28)))
                    if step % 1000 == 0 and step > 1:
                        saver.save(sess, FLAGS.model_dir + '/autoencoder-model', global_step=step)
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                coord.request_stop()
                coord.join(threads)


def train_autoencoder():
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)
    image_paths = FLAGS.train_image_dir
    ndim_z = FLAGS.ndim_z
    batchsize = FLAGS.batchsize
    input_image, _ = get_image(image_paths, 75, True)
    input_image = tf.reshape(input_image, [28, 28, 1])
    input_images = tf.train.batch([input_image], batchsize, dynamic_pad=True)
    # input_images = tf.reshape(input_images)
    # input_images, img_keys = tf.train.batch((input_image, img_key), batchsize, dynamic_pad=True)

    reconst, z_mean, z_log_sigma_sq, z = cvaemodel.conv_encoder_z_decoder(input_images, ndim_z)
    reconst_img = reconst * 255.0
    # flat_input = tf.to_float(tf.reshape(input_images, [batchsize, -1]))
    # flat_recon = tf.to_float(tf.reshape(reconst, [batchsize, -1]))
    # rec_loss = -tf.reduce_sum(flat_input * tf.log(1e-3 + flat_recon) + (1. - flat_input) * tf.log(1e-3 + 1. - flat_recon), 1)
    rec_loss = tf.reduce_sum(tf.square(input_images - reconst), [1, 2, 3])
    latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1)
    total_loss = tf.reduce_mean(rec_loss + latent_loss)
    rec_vars = [v for v in tf.all_variables() if v.name.startswith("encoder_z_decoder/")]
    rec_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    rec_step = tf.Variable(0, name="rec_step", trainable=False)
    rec_op = minimize_and_clip(rec_optimizer, objective=total_loss, var_list=rec_vars, step=rec_step)

    '''
    pz_priors = tf.placeholder(tf.float32, (batchsize, ndim_z))
    dis_latent_outputs = vaemodel.discriminator(latent)
    dis_prior_outputs = vaemodel.discriminator(pz_priors, reuse=True)

    dis_loss = tf.reduce_sum(tf.log(dis_prior_outputs) + tf.log(1. - dis_latent_outputs))
    dis_vars = [v for v in tf.all_variables() if v.name.startswith("discriminator/")]
    dis_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    dis_step = tf.Variable(0, name="dis_step", trainable=False)
    dis_op = minimize_and_clip(dis_optimizer, objective=-dis_loss, var_list=dis_vars, step=dis_step)

    gen_loss = tf.reduce_sum(tf.log(dis_latent_outputs))
    gen_vars = [v for v in tf.all_variables() if v.name.startswith("encoder_decoder/encoder/")]
    gen_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    gen_step = tf.Variable(0, name="gen_step", trainable=False)
    gen_op = minimize_and_clip(gen_optimizer, objective=-gen_loss, var_list=gen_vars, step=gen_step)
    '''

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
                    # aaa, bbb = sess.run([z_mean, z_log_sigma_sq])
                    totalloss, _, recstep = sess.run([total_loss, rec_op, rec_step])
                    # disloss, _, disstep = sess.run([dis_loss, dis_op, dis_step], {pz_priors: np.random.uniform(-2., 2., size=(batchsize, ndim_z)).astype(np.float32)})
                    # genloss, _, genstep = sess.run([gen_loss, gen_op, gen_step])
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    if recstep % 100 == 0:
                        print(recstep, totalloss, elapsed_time)
                        # print(disstep, disloss, elapsed_time)
                        # print(genstep, genstep, elapsed_time)
                        output_t = sess.run(reconst_img)
                        for i, raw_image in enumerate(output_t):
                            io.imsave('test/out%s-%s.png' % (recstep, i + 1), np.reshape(raw_image.astype("uint8"), (28, 28)))
                    if recstep % 1000 == 0:
                        saver.save(sess, FLAGS.model_dir + '/autoencoder-model', global_step=recstep)
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                coord.request_stop()
                coord.join(threads)


def test_autoencoder():
    image_paths = FLAGS.test_image_dir
    batchsize = FLAGS.batchsize
    input_image, img_key = get_image(image_paths, 1, False)
    input_image = tf.reshape(input_image, [28, 28, 1])
    input_images, img_keys = tf.train.batch((input_image, img_key), batchsize, dynamic_pad=True)
    # img_keys = tf.train.batch([img_key], batchsize, dynamic_pad=True)
    ndim_z = FLAGS.ndim_z
    collect_z = np.empty((1, FLAGS.ndim_z))
    collect_label = []
    reconst, z_mean, z_log_sigma_sq, z = cvaemodel.conv_encoder_z_decoder(input_images, ndim_z)
    reconst_img = tf.reshape(reconst, tf.shape(input_images)) * 255.0

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        sess.run(tf.initialize_local_variables())
        file = tf.train.latest_checkpoint(FLAGS.model_dir)
        if not file:
            print('Could not find trained model in {}'.format(FLAGS.model_dir))
            return
        print('Using model from {}'.format(file))
        saver = tf.train.Saver()
        saver.restore(sess, file)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        start_time = time.time()
        try:
            for step in xrange(1000000):
                if coord.should_stop():
                    break
                print(step)
                output_t, latent, output_n = sess.run([reconst_img, z, img_keys])
                collect_z = np.r_[collect_z, latent]

                elapsed = time.time() - start_time
                start_time = time.time()
                print('Time for one batch: {}'.format(elapsed))
                for i, raw_image in enumerate(output_t):
                    filepath = output_n[i]
                    filename = filepath[filepath.find("/") + 1:filepath.find(".")]
                    label = filename[0]
                    collect_label.append(int(label[0]))
                    io.imsave('test/%s_%s.png' % ('rec', filename), np.reshape(raw_image.astype("uint8"), (28, 28)))
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
            coord.join(threads)
    return collect_z, collect_label


def plot_labeled_z(collect_z, collect_label, dir='.', filename="labeled_z"):
    plt.figure(figsize=(8, 6))
    plt.scatter(collect_z[1:, 0], collect_z[1:, 1], c=collect_label, s=10, marker="o")
    plt.colorbar()
    plt.savefig("{}/{}.png".format(dir, filename))


def plot_canvas(reconst):
    nx = ny = 20
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)

    canvas = np.empty((28 * ny, 28 * nx))
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            z_mu = np.array([[xi, yi]])
            with tf.Session() as sess:
                output_t = sess.run(reconst, feed_dict={z: z_mu})
            canvas[(nx - i - 1) * 28:(nx - i) * 28, j * 28:(j + 1) * 28] = output_t[0].reshape(28, 28)

    plt.figure(figsize=(8, 10))
    plt.colorbar()
    Xi, Yi = np.meshgrid(x_values, y_values)
    plt.imshow(canvas, origin="upper")
    plt.tight_layout()


def main():
    collect_z, collect_label = test_autoencoder()
    # plot_labeled_z(collect_z, collect_label, dir='.', filename="labeled_z")


if __name__ == '__main__':
    # train_autoencoder()
    # main()
    train_gan()
