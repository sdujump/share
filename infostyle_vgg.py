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


batch_size = FLAGS.batch_size
with h5py.File(''.join(['datasets/dataset-rgb-32.h5']), 'r') as hf:
    images = hf['images'].value
    filenames = hf['filenames'].value

# iter_ = infostyle_util.data_iterator(images, filenames, batch_size)
# image, filename = iter_.next()


def show_variables(variales):
    for i in range(len(variales)):
        print(variales[i].name)


# tf.reset_default_graph()
layer = 'relu2_2'
num_gpus = 1
z_size = 64  # Size of initial z vector used for generator.
image_size = 32
# Define latent variables.
# Each entry in this list defines a categorical variable of a specific size.
categorical_list = 10
number_continuous = 2  # The number of continous variables.
num_examples = 60000
num_epochs = 50  # Total number of iterations to use.
total_batch = int(np.floor(num_examples / (batch_size)))
# Directory to save sample images from generator in.
sample_directory = './figsTut'
model_directory = './models'  # Directory to save trained model to.
mean_pixel = [123.68, 116.779, 103.939]  # ImageNet average from VGG ..


def train_infogan():

    style_names = [join('styles', f) for f in listdir('styles') if isfile(join('styles', f)) & f.lower().endswith('jpg')]
    style_image = scipy.misc.imread(style_names[0], mode='RGB') - mean_pixel
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    # global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    # This initializaer is used to initialize all the weights of the network.
    # initializer = tf.truncated_normal_initializer(stddev=0.02)

    # These placeholders are used for input into the generator and discriminator, respectively.
    style_holder = tf.placeholder(shape=[1, None, None, 3], dtype=tf.float32)  # Real images
    real_in_list = tf.placeholder(shape=[None, image_size, image_size, 3], dtype=tf.float32)  # Real images
    z_lat = tf.placeholder(shape=[None, z_size + categorical_list + number_continuous], dtype=tf.float32)  # Random vector
    style_net, _ = vgg.net(FLAGS.VGG_PATH, style_holder)

    # The below code is responsible for applying gradient descent to update
    # the GAN.
    trainerD = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
    trainerG = tf.train.AdamOptimizer(learning_rate=0.002, beta1=0.5)
    trainerQ = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)

    d_loss, g_loss, q_loss, Gz = tower_loss(real_in_list, z_lat)

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
        iter_ = infostyle_util.data_iterator(images, filenames, batch_size)
        for i in tqdm.tqdm(xrange(total_batch)):

            # Concatenate all c and z variables.
            zlat = latent_prior(z_size, batch_size, 1)

            # Draw a sample batch from MNIST dataset.
            xs, _ = iter_.next()
            xs = (np.reshape(xs, [batch_size, image_size, image_size, 3]) / 255.0 - 0.5) * 2.0

            _, dLoss = sess.run([update_D, d_loss], feed_dict={real_in_list: xs, z_lat: zlat})  # Update the discriminator
            # Update the generator, twice for good measure.
            _, gLoss = sess.run([update_G, g_loss], feed_dict={z_lat: zlat})
            _, qLoss = sess.run([update_Q, q_loss], feed_dict={z_lat: zlat})  # Update to optimize mutual information.

            if i % 50 == 0:
                print "epoch: " + str(epoch) + " Gen Loss: " + str(gLoss) + " Disc Loss: " + str(dLoss) + " Q Losses: " + str(qLoss)
                # Generate another z batch
                z_sample = np.random.uniform(-1.0, 1.0, size=[batch_size, z_size]).astype(np.float32)
                lcat_sample = np.array([e for e in range(10) for tempi in range(10)])
                latent_oh = np.zeros((batch_size, 10))
                latent_oh[np.arange(batch_size), lcat_sample] = 1

                aa = np.reshape(np.array([[(ee / 4.5 - 1)] for ee in range(10) for tempj in range(10)]), [10, 10]).T
                bb = np.reshape(aa, [batch_size, 1])
                cc = np.zeros_like(bb)
                lcont_sample = np.hstack([bb, cc])

                # Concatenate all c and z variables.
                zlat = np.concatenate([latent_oh, z_sample, lcont_sample], 1).astype(np.float32)
                zlats = np.concatenate([zlat, zlat, zlat, zlat], 0).astype(np.float32)
                # Use new z to get sample images from generator.
                samples = sess.run(Gz, feed_dict={z_lat: zlats})
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


def tower_loss(real_in, z_lat):

    Gz = infostyle_util.generator(z_lat)  # Generates images from random z vectors
    # Produces probabilities for real images
    Dx, _, _ = infostyle_util.discriminator(real_in)
    # Produces probabilities for generator images
    Dg, QgCat, QgCont = infostyle_util.discriminator(Gz, reuse=True)

    # These functions together define the optimization objective of the GAN.
    # This optimizes the discriminator.
    d_loss = -tf.reduce_mean(tf.log(Dx) + tf.log(1. - Dg), name='dloss')
    g_loss = -tf.reduce_mean(tf.log((Dg / (1 - Dg))), name='gloss')  # KL Divergence optimizer
    # d_loss = tf.Print(d_loss, [d_loss], 'd_loss = ', summarize=-1)

    # Combine losses for each of the categorical variables.
    cat_losses = []
    cat_loss = -tf.reduce_sum(z_lat[:, 0:categorical_list] * tf.log(QgCat[0]), reduction_indices=1)
    cat_losses.append(cat_loss)

    # Combine losses for each of the continous variables.
    q_cont_loss = tf.reduce_sum(0.5 * tf.square(z_lat[:, categorical_list + z_size:] - QgCont), reduction_indices=1)

    q_cont_loss = tf.reduce_mean(q_cont_loss)
    q_cat_loss = tf.reduce_mean(cat_losses)
    q_loss = tf.add(q_cat_loss, q_cont_loss, name='qloss')

    return d_loss, g_loss, q_loss, Gz


def latent_prior(z_size, batch_size, style_image, num_gpus):

    style_net, _ = vgg.net(FLAGS.VGG_PATH, style_image)
    style_layer = style_net[layer]
    z_gram = infostyle_util.gram_encoder(infostyle_util.gram_np(style_layer))
    z_grams = np.repeat(z_gram, batch_size, 0)
    zs = np.random.uniform(-1.0, 1.0, size=[batch_size * num_gpus, z_size]).astype(np.float32)

    # Concatenate all c and z variables.
    zlat = np.concatenate([zs, z_grams], 1).astype(np.float32)
    return zlat


if __name__ == '__main__':
    train_infogan()
    # test_infogan()
