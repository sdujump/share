# Import the libraries we will need.
import tensorflow as tf
import numpy as np
import os
import infostyle_util
import h5py  # for reading our dataset
from tensorflow.python.client import device_lib
import tqdm  # making loops prettier
import vgg
from os import listdir
from os.path import isfile, join

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print device_lib.list_local_devices()

tf.app.flags.DEFINE_string("train_image_dir", "train_images", "")
tf.app.flags.DEFINE_string("test_image_dir", "test_images", "")
tf.app.flags.DEFINE_string("model_dir", "model2", "")
tf.app.flags.DEFINE_integer("batch_size", 100, "")
FLAGS = tf.app.flags.FLAGS


with h5py.File(''.join(['datasets/coco-256.h5']), 'r') as hf:
    real_images = hf['images'].value
    real_names = hf['filenames'].value

with h5py.File(''.join(['datasets/coco-256.h5']), 'r') as hf:
    style_images = hf['images'].value
    style_names = hf['filenames'].value
    style_grams = hf['grams'].value


def show_variables(variales):
    for i in range(len(variales)):
        print(variales[i].name)


mean_pixel = [123.68, 116.779, 103.939]  # ImageNet average from VGG ..
layer = 'relu2_2'
num_gpus = 1
z_size = 128  # Size of initial z vector used for generator.
image_size = 128
# Define latent variables.
# Each entry in this list defines a categorical variable of a specific size.
num_examples = 60000
num_epochs = 50  # Total number of iterations to use.
batch_size = FLAGS.batch_size
total_batch = int(np.floor(num_examples / (batch_size)))
# Directory to save sample images from generator in.
sample_directory = './figsTut'
model_directory = './models'  # Directory to save trained model to.
mean_pixel = [123.68, 116.779, 103.939]  # ImageNet average from VGG ..


def train_infogan():

    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    # global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    # This initializaer is used to initialize all the weights of the network.
    # initializer = tf.truncated_normal_initializer(stddev=0.02)

    # These placeholders are used for input into the generator and discriminator, respectively.
    real_in_list = tf.placeholder(shape=[None, image_size, image_size, 3], dtype=tf.float32)  # Real images
    gram_in_list = tf.placeholder(shape=[None, 256], dtype=tf.float32)
    Z_s = tf.placeholder(shape=[None, z_size], dtype=tf.float32)  # Random vector

    # The below code is responsible for applying gradient descent to update
    # the GAN.
    trainerD = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
    trainerG = tf.train.AdamOptimizer(learning_rate=0.002, beta1=0.5)
    trainerQ = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)

    d_loss, g_loss, q_loss, Gz = tower_loss(real_in_list, gram_in_list, Z_s)

    tvars = tf.trainable_variables()
    gen_vars = [v for v in tvars if v.name.startswith("generator/")]
    dis_vars = [v for v in tvars if v.name.startswith("discriminator/")]

    tf.get_variable_scope().reuse_variables()  # important

    # Only update the weights for the discriminator network.
    d_grads = trainerD.compute_gradients(d_loss, dis_vars)
    # Only update the weights for the generator network.
    g_grads = trainerG.compute_gradients(g_loss, gen_vars)
    q_grads = trainerG.compute_gradients(q_loss, tvars)

    update_D = trainerD.apply_gradients(d_grads)
    update_G = trainerG.apply_gradients(g_grads)
    update_Q = trainerQ.apply_gradients(q_grads)

    saver = tf.train.Saver(tf.all_variables())
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    epoch = 0

    while epoch < num_epochs:
        real_iter_ = infostyle_util.data_iterator(real_images, real_names, batch_size)
        for i in tqdm.tqdm(xrange(total_batch)):

            real_batch, _ = real_iter_.next()
            real_batch = (np.reshape(real_batch, [batch_size, image_size, image_size, 3]) / 255.0 - 0.5) * 2.0

            style_iter_ = infostyle_util.data_iterator(style_images, style_names, style_grams, batch_size)
            _, _, gram_batch = style_iter_.next()

            # Concatenate all c and z variables.
            # zlat = latent_prior(z_size, batch_size, 1)
            zs = np.random.uniform(-1.0, 1.0, size=[batch_size, z_size]).astype(np.float32)
            # Draw a sample batch from MNIST dataset.

            _, dLoss = sess.run([update_D, d_loss], feed_dict={real_in_list: real_batch, gram_in_list: gram_batch, Z_s: zs})  # Update the discriminator
            # Update the generator, twice for good measure.
            _, gLoss = sess.run([update_G, g_loss], feed_dict={gram_in_list: gram_batch, Z_s: zs})
            _, qLoss = sess.run([update_Q, q_loss], feed_dict={gram_in_list: gram_batch, Z_s: zs})  # Update to optimize mutual information.

            if i % 50 == 0:
                print "epoch: " + str(epoch) + " Gen Loss: " + str(gLoss) + " Disc Loss: " + str(dLoss) + " Q Losses: " + str(qLoss)
                # Generate another z batch
                # Use new z to get sample images from generator.
                samples = sess.run(Gz, feed_dict={gram_in_list: gram_batch, Z_s: zs})
                if not os.path.exists(sample_directory):
                    os.makedirs(sample_directory)
                # Save sample generator images for viewing training
                # progress.
                infostyle_util.save_images(np.reshape(samples[0:batch_size], [batch_size, image_size, image_size, 3]), [10, 10], sample_directory + '/fig' + str(epoch) + str(i) + '.png')
        if epoch % 3 == 0:
            if not os.path.exists(model_directory):
                os.makedirs(model_directory)
            saver.save(sess, model_directory + '/model-epoch-' + str(epoch) + '.cptk', global_step=epoch)
            print "Saved Model"
        epoch += 1


def tower_loss(real_in, gram_in, zs):

    # gram = infostyle_util.gram_np(style_in)
    z_gram = infostyle_util.gram_encoder(gram_in)
    z_lat = tf.concat([zs, z_gram], 1)

    Gz = infostyle_util.generator(z_lat)  # Generates images from random z vectors
    # Produces probabilities for real images
    Dx, _, _ = infostyle_util.discriminator(real_in)
    # Produces probabilities for generator images
    Dg, QgGram = infostyle_util.discriminator(Gz, reuse=True)

    # These functions together define the optimization objective of the GAN.
    # This optimizes the discriminator.
    d_loss = -tf.reduce_mean(tf.log(Dx) + tf.log(1. - Dg), name='dloss')
    g_loss = -tf.reduce_mean(tf.log((Dg / (1 - Dg))), name='gloss')  # KL Divergence optimizer

    # Combine losses for each of the continous variables.
    gram_losses = tf.reduce_sum(0.5 * tf.square(gram_in - QgGram), reduction_indices=1)
    q_loss = tf.reduce_mean(gram_losses, name='qloss')

    return d_loss, g_loss, q_loss, Gz


if __name__ == '__main__':
    train_infogan()
    # test_infogan()
