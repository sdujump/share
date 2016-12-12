# Import the libraries we will need.
import tensorflow as tf
import numpy as np
import os
import gram_util
import h5py  # for reading our dataset
from tensorflow.python.client import device_lib
import tqdm  # making loops prettier
import vgg
from os import listdir
from os.path import isfile, join

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print device_lib.list_local_devices()


with h5py.File(''.join(['datasets/dataset-grams-relu3_4.h5']), 'r') as hf:
    grams = hf['grams'].value
    filenames = hf['filenames'].value

def show_variables(variales):
    for i in range(len(variales)):
        print(variales[i].name)


mean_pixel = [123.68, 116.779, 103.939]  # ImageNet average from VGG ..
layer = 'relu3_4'
z_size = 128  # Size of initial z vector used for generator.
image_size = 128
# Define latent variables.
# Each entry in this list defines a categorical variable of a specific size.
num_epochs = 100  # Total number of iterations to use.
batch_size = 10
total_batch = 40
# Directory to save sample images from generator in.
sample_directory = './figsTut'
model_directory = './models'  # Directory to save trained model to.
mean_pixel = [123.68, 116.779, 103.939]  # ImageNet average from VGG ..


def train_infogan():
    # These placeholders are used for input into the generator and discriminator, respectively.
    gram_in = tf.placeholder(shape=[batch_size, 32896], dtype=tf.float32)
    Z_s = tf.placeholder(shape=[batch_size, z_size], dtype=tf.float32)  # Random vector

    # The below code is responsible for applying gradient descent to update
    # the GAN.
    trainerQ = tf.train.AdamOptimizer(learning_rate=0.002, beta1=0.5)

    q_loss, Gz = sanity_loss(gram_in, Z_s)

    tvars = tf.trainable_variables()

    # Only update the weights for the discriminator network.
    q_grads = trainerQ.compute_gradients(q_loss, tvars)

    update_Q = trainerQ.apply_gradients(q_grads)

    saver = tf.train.Saver(tf.all_variables())
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    epoch = 0

    while epoch < num_epochs:
        style_iter_ = gram_util.data_iterator(grams, filenames, 1)
        for i in tqdm.tqdm(xrange(total_batch)):
            gram, gram_name = style_iter_.next()
            gram_batch = np.repeat(gram, batch_size, 0)
            # Concatenate all c and z variables.
            # zlat = latent_prior(z_size, batch_size, 1)
            zs = np.random.uniform(-1.0, 1.0, size=[batch_size, z_size]).astype(np.float32)
            # Draw a sample batch from MNIST dataset.

            _, qLoss = sess.run([update_Q, q_loss], feed_dict={gram_in: gram_batch, Z_s: zs})  # Update to optimize mutual information.
            print "epoch: " + str(epoch) + " Q Losses: " + str(qLoss)

            if i % 39 == 0:
                # Generate another z batch
                # Use new z to get sample images from generator.
                samples = sess.run(Gz, feed_dict={gram_in: gram_batch, Z_s: zs})
                if not os.path.exists(sample_directory):
                    os.makedirs(sample_directory)
                # Save sample generator images for viewing training
                # progress.
                gram_util.save_images(np.reshape(samples[0:batch_size], [batch_size, image_size, image_size, 3]), [1, 10], sample_directory + '/fig' + str(epoch) + str(i) + '.png')
        if epoch % 3 == 0:
            if not os.path.exists(model_directory):
                os.makedirs(model_directory)
            saver.save(sess, model_directory + '/model-epoch-' + str(epoch) + '.cptk', global_step=epoch)
            print "Saved Model"
        epoch += 1


def sanity_loss(gram_in, zs):

    z_gram = gram_util.gram_encoder(gram_in)
    z_lat = tf.concat(1, [zs, z_gram])

    Gz = gram_util.generator(z_lat)  # Generates images from random z vectors
    # Produces probabilities for real images

    x_net, _ = vgg.net('imagenet-vgg-verydeep-19.mat', Gz)

    x_layer = x_net['relu3_4']
    gram_out = gram_util.gram(x_layer)

    # This optimizes the discriminator.
    # Combine losses for each of the continous variables.
    gram_losses = tf.reduce_sum(0.5 * tf.square(gram_in[0] - gram_out), reduction_indices=0)

    return gram_losses, Gz


if __name__ == '__main__':
    train_infogan()
    # test_infogan()
