# Import the libraries we will need.
import tensorflow as tf
import numpy as np
import os
import scipy.misc
import infostyle_util3 as infostyle_util
import h5py  # for reading our dataset
from tensorflow.python.client import device_lib
import tqdm  # making loops prettier

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print device_lib.list_local_devices()

tf.app.flags.DEFINE_string("train_image_dir", "train_images", "")
tf.app.flags.DEFINE_string("test_image_dir", "test_images", "")
tf.app.flags.DEFINE_string("model_dir", "model2", "")
tf.app.flags.DEFINE_integer("batch_size", 100, "")
FLAGS = tf.app.flags.FLAGS


batch_size = FLAGS.batch_size
with h5py.File(''.join(['datasets/dataset-rgb-32.h5']), 'r') as hf:
    images = hf['images'].value
    filenames = hf['filenames'].value

# iter_ = infostyle_util.data_iterator(images, filenames, batch_size)
# image, filename = iter_.next()


def show_variables(variales):
    for i in range(len(variales)):
        print(variales[i].name)


tf.reset_default_graph()

num_gpus = 1
z_size = 64  # Size of initial z vector used for generator.
image_size = 32
# Define latent variables.
# Each entry in this list defines a categorical variable of a specific size.
categorical_list = 10
number_continuous = 2  # The number of continous variables.

# This initializaer is used to initialize all the weights of the network.
initializer = tf.truncated_normal_initializer(stddev=0.02)

# These placeholders are used for input into the generator and discriminator, respectively.
real_in = tf.placeholder(shape=[None, image_size, image_size, 3], dtype=tf.float32)  # Real images

z_lat = tf.placeholder(shape=[None, z_size + categorical_list + number_continuous], dtype=tf.float32)  # Random vector


# The below code is responsible for applying gradient descent to update
# the GAN.
trainerD = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
trainerG = tf.train.AdamOptimizer(learning_rate=0.002, beta1=0.5)
trainerQ = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)

tower_grads_D = []
tower_grads_G = []
tower_grads_Q = []
Output = []

for i in xrange(num_gpus):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('Tower_%d' % (i)) as scope:

                Gz = infostyle_util.generator(z_lat[i * batch_size:(i + 1) * batch_size, :])  # Generates images from random z vectors
                # Produces probabilities for real images
                Dx, _, _ = infostyle_util.discriminator(real_in[i * batch_size:(i + 1) * batch_size, :])
                # Produces probabilities for generator images
                Dg, QgCat, QgCont = infostyle_util.discriminator(Gz, reuse=True)

                # These functions together define the optimization objective of the GAN.
                # This optimizes the discriminator.
                d_loss = -tf.reduce_mean(tf.log(Dx) + tf.log(1. - Dg))
                g_loss = -tf.reduce_mean(tf.log((Dg / (1 - Dg))))  # KL Divergence optimizer

                # Combine losses for each of the categorical variables.
                cat_losses = []
                cat_loss = -tf.reduce_sum(z_lat[i * batch_size:(i + 1) * batch_size, 0:categorical_list] * tf.log(QgCat[0]), reduction_indices=1)
                cat_losses.append(cat_loss)

                # Combine losses for each of the continous variables.
                q_cont_loss = tf.reduce_sum(0.5 * tf.square(z_lat[i * batch_size:(i + 1) * batch_size, categorical_list + z_size:] - QgCont), reduction_indices=1)

                q_cont_loss = tf.reduce_mean(q_cont_loss)
                q_cat_loss = tf.reduce_mean(cat_losses)
                q_loss = tf.add(q_cat_loss, q_cont_loss)
                tvars = tf.trainable_variables()

                gen_vars = [v for v in tvars if v.name.startswith("generator/")]
                dis_vars = [v for v in tvars if v.name.startswith("discriminator/")]

                tf.get_variable_scope().reuse_variables()  # important

                # Only update the weights for the discriminator network.
                d_grads = trainerD.compute_gradients(d_loss, dis_vars)
                # Only update the weights for the generator network.
                g_grads = trainerG.compute_gradients(g_loss, gen_vars)
                q_grads = trainerG.compute_gradients(q_loss, tvars)

                # Keep track of the gradients across all towers.
                tower_grads_D.append(d_grads)
                tower_grads_G.append(g_grads)
                tower_grads_Q.append(q_grads)
                Output.append(Gz)

# Average the gradients
grads_d = infostyle_util.average_gradients(tower_grads_D)
grads_g = infostyle_util.average_gradients(tower_grads_G)
grads_q = infostyle_util.average_gradients(tower_grads_Q)

update_D = trainerD.apply_gradients(grads_d)
update_G = trainerG.apply_gradients(grads_g)
update_Q = trainerQ.apply_gradients(grads_q)

Outputs = tf.concat(1, Output)


def train_infogan():
    num_examples = 60000
    epoch = 0
    num_epochs = 50  # Total number of iterations to use.
    total_batch = int(np.floor(num_examples / (batch_size * num_gpus)))
    # Directory to save sample images from generator in.
    sample_directory = './figsTut'
    model_directory = './models'  # Directory to save trained model to.
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(init)
        while epoch < num_epochs:
            iter_ = infostyle_util.data_iterator(images, filenames, batch_size * num_gpus)
            for i in tqdm.tqdm(range(total_batch)):
                # Generate a random z batch
                zs = np.random.uniform(-1.0, 1.0, size=[batch_size * num_gpus, z_size]).astype(np.float32)
                lcont = np.random.uniform(-1, 1, [batch_size * num_gpus, number_continuous])

                lcat = np.random.randint(0, 10, [batch_size * num_gpus, ])  # Generate random c batch
                latent_oh = np.zeros((batch_size * num_gpus, 10))
                latent_oh[np.arange(batch_size * num_gpus), lcat] = 1

                # Concatenate all c and z variables.
                zlat = np.concatenate([latent_oh, zs, lcont], 1).astype(np.float32)

                # Draw a sample batch from MNIST dataset.
                xs, _ = iter_.next()
                # xs, _ = mnist.train.next_batch(batch_size)
                # Transform it to be between -1 and 1
                xs = (np.reshape(xs, [batch_size * num_gpus, image_size, image_size, 3]) / 255.0 - 0.5) * 2.0
                # xs = np.lib.pad(xs, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant', constant_values=(-1, -1))  # Pad the images so the are 32x32

                _, dLoss = sess.run([update_D, d_loss], feed_dict={z_lat: zlat, real_in: xs})  # Update the discriminator
                # Update the generator, twice for good measure.
                _, gLoss = sess.run([update_G, g_loss], feed_dict={z_lat: zlat})
                _, qLoss, qK, qC = sess.run([update_Q, q_loss, q_cont_loss, q_cat_loss], feed_dict={z_lat: zlat})  # Update to optimize mutual information.

                if i % 100 == 0:
                    print "epoch: " + str(epoch) + " Gen Loss: " + str(gLoss) + " Disc Loss: " + str(dLoss) + " Q Losses: " + str([qK, qC])
                    # Generate another z batch
                    z_sample = np.random.uniform(-1.0, 1.0, size=[batch_size * num_gpus, z_size]).astype(np.float32)
                    lcat_sample = np.array([e for e in range(10) for tempi in range(10 * num_gpus)])
                    latent_oh = np.zeros((batch_size * num_gpus, 10))
                    latent_oh[np.arange(batch_size * num_gpus), lcat_sample] = 1

                    tempg = np.sqrt(num_gpus)
                    aa = np.reshape(np.array([[(ee / 4.5 - tempg)] for ee in range(10 * tempg) for tempj in range(10 * tempg)]), [10 * tempg, 10 * tempg]).T
                    bb = np.reshape(aa, [batch_size * num_gpus, 1])
                    cc = np.zeros_like(bb)
                    lcont_sample = np.hstack([bb, cc])

                    # Concatenate all c and z variables.
                    zlat = np.concatenate([latent_oh, z_sample, lcont_sample], 1).astype(np.float32)
                    # Use new z to get sample images from generator.
                    samples = sess.run(Outputs, feed_dict={z_lat: zlat})
                    if not os.path.exists(sample_directory):
                        os.makedirs(sample_directory)
                    # Save sample generator images for viewing training
                    # progress.
                    infostyle_util.save_images(np.reshape(samples[0:batch_size * num_gpus], [batch_size * num_gpus, image_size, image_size, 3]), [10 * tempg, 10 * tempg], sample_directory + '/fig' + str(epoch) + str(i) + '.png')
            epoch += 1
            if not os.path.exists(model_directory):
                os.makedirs(model_directory)
            saver.save(sess, model_directory + '/model-epoch-' + str(epoch) + '.cptk')
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
        z_sample = np.random.uniform(-1.0, 1.0, size=[batch_size, z_size]).astype(np.float32)
        lcat_sample = np.array([e for e in range(10) for tempi in range(10)])
        latent_oh = np.zeros((batch_size, 10))
        latent_oh[np.arange(batch_size), lcat_sample] = 1

        aa = np.reshape(np.array([[(ee / 4.5 - 1.)] for ee in range(10) for tempj in range(10)]), [10, 10]).T
        bb = np.reshape(aa, [100, 1])
        cc = np.zeros_like(bb)
        lcont_sample = np.hstack([bb, cc])

        # Concatenate all c and z variables.
        zlat = np.concatenate([latent_oh, z_sample, lcont_sample], 1).astype(np.float32)
        # Use new z to get sample images from generator.
        samples = sess.run(Gz, feed_dict={z_lat: zlat})
        if not os.path.exists(sample_directory):
            os.makedirs(sample_directory)
        # Save sample generator images for viewing training progress.
        infostyle_util.save_images(np.reshape(samples[0:100], [batch_size, image_size, image_size]), [10, 10], sample_directory + '/fig_test' + '.png')


if __name__ == '__main__':
    train_infogan()
    # test_infogan()
