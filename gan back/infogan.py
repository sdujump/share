# Import the libraries we will need.
import tensorflow as tf
import numpy as np
import input_data
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import os
import scipy.misc
import scipy
import infogan_util


mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)


tf.reset_default_graph()

z_size = 64  # Size of initial z vector used for generator.

# Define latent variables.
# Each entry in this list defines a categorical variable of a specific size.
categorical_list = [10]
number_continuous = 2  # The number of continous variables.

# This initializaer is used to initialize all the weights of the network.
initializer = tf.truncated_normal_initializer(stddev=0.02)

# These placeholders are used for input into the generator and discriminator, respectively.
z_in = tf.placeholder(shape=[None, z_size], dtype=tf.float32)  # Random vector
real_in = tf.placeholder(shape=[None, 32, 32, 1], dtype=tf.float32)  # Real images

# These placeholders load the latent variables.
latent_cat_in = tf.placeholder(shape=[None, len(categorical_list)], dtype=tf.int32)
latent_cat_list = tf.split(1, len(categorical_list), latent_cat_in)
latent_cont_in = tf.placeholder(shape=[None, number_continuous], dtype=tf.float32)

oh_list = []
for idx, var in enumerate(categorical_list):
    latent_oh = tf.one_hot(tf.reshape(latent_cat_list[idx], [-1]), var)
    oh_list.append(latent_oh)

# Concatenate all c and z variables.
z_lats = oh_list[:]
z_lats.append(z_in)
z_lats.append(latent_cont_in)
z_lat = tf.concat(1, z_lats)


Gz = infogan_util.generator(z_lat)  # Generates images from random z vectors
# Produces probabilities for real images
Dx, _, _ = infogan_util.discriminator(real_in, categorical_list, number_continuous)
# Produces probabilities for generator images
Dg, QgCat, QgCont = infogan_util.discriminator(
    Gz, categorical_list, number_continuous, reuse=True)

# These functions together define the optimization objective of the GAN.
# This optimizes the discriminator.
d_loss = -tf.reduce_mean(tf.log(Dx) + tf.log(1. - Dg))
g_loss = -tf.reduce_mean(tf.log((Dg / (1 - Dg))))  # KL Divergence optimizer

# Combine losses for each of the categorical variables.
cat_losses = []
for idx, latent_var in enumerate(oh_list):
    cat_loss = -tf.reduce_sum(latent_var * tf.log(QgCat[idx]), reduction_indices=1)
    cat_losses.append(cat_loss)

# Combine losses for each of the continous variables.
if number_continuous > 0:
    q_cont_loss = tf.reduce_sum(
        0.5 * tf.square(latent_cont_in - QgCont), reduction_indices=1)
else:
    q_cont_loss = tf.constant(0.0)

q_cont_loss = tf.reduce_mean(q_cont_loss)
q_cat_loss = tf.reduce_mean(cat_losses)
q_loss = tf.add(q_cat_loss, q_cont_loss)
tvars = tf.trainable_variables()

# The below code is responsible for applying gradient descent to update
# the GAN.
trainerD = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
trainerG = tf.train.AdamOptimizer(learning_rate=0.002, beta1=0.5)
trainerQ = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
# Only update the weights for the discriminator network.
d_grads = trainerD.compute_gradients(
    d_loss, tvars[9:-2 - ((number_continuous > 0) * 2) - (len(categorical_list) * 2)])
# Only update the weights for the generator network.
g_grads = trainerG.compute_gradients(g_loss, tvars[0:9])
q_grads = trainerG.compute_gradients(q_loss, tvars)

update_D = trainerD.apply_gradients(d_grads)
update_G = trainerG.apply_gradients(g_grads)
update_Q = trainerQ.apply_gradients(q_grads)


def train_infogan():
    batch_size = 64  # Size of image batch to apply at each iteration.
    iterations = 500000  # Total number of iterations to use.
    # Directory to save sample images from generator in.
    sample_directory = './figsTut'
    model_directory = './models'  # Directory to save trained model to.
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations):
            # Generate a random z batch
            zs = np.random.uniform(-1.0, 1.0, size=[batch_size, z_size]).astype(np.float32)
            lcat = np.random.randint(0, 10, [batch_size, len(categorical_list)])  # Generate random c batch
            lcont = np.random.uniform(-1, 1, [batch_size, number_continuous])

            # Draw a sample batch from MNIST dataset.
            xs, _ = mnist.train.next_batch(batch_size)
            # Transform it to be between -1 and 1
            xs = (np.reshape(xs, [batch_size, 28, 28, 1]) - 0.5) * 2.0
            xs = np.lib.pad(xs, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant',
                            constant_values=(-1, -1))  # Pad the images so the are 32x32

            _, dLoss = sess.run([update_D, d_loss], feed_dict={
                                z_in: zs, real_in: xs, latent_cat_in: lcat, latent_cont_in: lcont})  # Update the discriminator
            # Update the generator, twice for good measure.
            _, gLoss = sess.run([update_G, g_loss], feed_dict={
                                z_in: zs, latent_cat_in: lcat, latent_cont_in: lcont})
            _, qLoss, qK, qC = sess.run([update_Q, q_loss, q_cont_loss, q_cat_loss], feed_dict={
                                        z_in: zs, latent_cat_in: lcat, latent_cont_in: lcont})  # Update to optimize mutual information.
            if i % 100 == 0:
                print "Gen Loss: " + str(gLoss) + " Disc Loss: " + str(dLoss) + " Q Losses: " + str([qK, qC])
                # Generate another z batch
                z_sample = np.random.uniform(-1.0, 1.0,
                                             size=[100, z_size]).astype(np.float32)
                lcat_sample = np.reshape(np.array([e for e in range(10) for tempi in range(10)]), [100, 1])
                a = a = np.reshape(
                    np.array([[(e / 4.5 - 1.)] for e in range(10) for tempj in range(10)]), [10, 10]).T
                b = np.reshape(a, [100, 1])
                c = np.zeros_like(b)
                lcont_sample = np.hstack([b, c])
                # Use new z to get sample images from generator.
                samples = sess.run(Gz, feed_dict={z_in: z_sample, latent_cat_in: lcat_sample, latent_cont_in: lcont_sample})
                if not os.path.exists(sample_directory):
                    os.makedirs(sample_directory)
                # Save sample generator images for viewing training
                # progress.
                infogan_util.save_images(np.reshape(samples[0:100], [100, 32, 32]), [10, 10], sample_directory + '/fig' + str(i) + '.png')
            if i % 1000 == 0 and i != 0:
                if not os.path.exists(model_directory):
                    os.makedirs(model_directory)
                saver.save(sess, model_directory + '/model-' + str(i) + '.cptk')
                print "Saved Model"


def test_infogan():
    # Directory to save sample images from generator in.
    sample_directory = './figsTut'
    model_directory = './models'  # Directory to load trained model from.

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        # Reload the model.
        print 'Loading Model...'
        ckpt = tf.train.get_checkpoint_state(model_directory)
        saver.restore(sess, ckpt.model_checkpoint_path)

        # Generate another z batch
        z_sample = np.random.uniform(-1.0, 1.0,
                                     size=[100, z_size]).astype(np.float32)
        lcat_sample = np.reshape(
            np.array([e for e in range(10) for _ in range(10)]), [100, 1])
        a = a = np.reshape(np.array([[(e / 4.5 - 1.)] for e in range(10) for _ in range(10)]), [10, 10]).T
        b = np.reshape(a, [100, 1])
        c = np.zeros_like(b)
        lcont_sample = np.hstack([b, c])
        # Use new z to get sample images from generator.
        samples = sess.run(Gz, feed_dict={
                           z_in: z_sample, latent_cat_in: lcat_sample, latent_cont_in: lcont_sample})
        if not os.path.exists(sample_directory):
            os.makedirs(sample_directory)
        # Save sample generator images for viewing training progress.
        infogan_util.save_images(np.reshape(samples[0:100], [100, 32, 32]), [10, 10], sample_directory + '/fig_test' + '.png')


if __name__ == '__main__':
    train_infogan()



