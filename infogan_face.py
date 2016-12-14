# Import the libraries we will need.
import tensorflow as tf
import numpy as np
import os
import infogan_faceutil
import h5py  # for reading our dataset
from tensorflow.python.client import device_lib
import tqdm  # making loops prettier
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print device_lib.list_local_devices()

FLAGS = tf.app.flags.FLAGS


hf = h5py.File(''.join(['/home/jump/data/img_align_celeba.h5']), 'r')
images = hf['images']
filenames = hf['filenames']

# image, filename = iter_.next()


def center_crop(x, crop_h, crop_w=None):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[1:3]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    croped = x[:, j:j + crop_h, i:i + crop_w, :]
    # sess = tf.InteractiveSession()
    # tf_resized = tf.image.resize_images(croped, [resize_w, resize_w])
    # resized = tf_resized.eval()
    # sess.close()
    return croped


def show_variables(variales):
    for i in range(len(variales)):
        print(variales[i].name)


# tf.reset_default_graph()
batch_size = 100
z_size = 100  # Size of initial z vector used for generator.
input_size = 108
image_size = 64
# Define latent variables.
# Each entry in this list defines a categorical variable of a specific size.
# categorical_list = 10
number_continuous = 2  # The number of continous variables.
num_examples = images.shape[0]
num_epochs = 50  # Total number of iterations to use.
total_batch = int(np.floor(num_examples / (batch_size)))
# Directory to save sample images from generator in.
sample_directory = './figsTut'
model_directory = './models'  # Directory to save trained model to.


def train_infogan():
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    # global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    # This initializaer is used to initialize all the weights of the network.
    # initializer = tf.truncated_normal_initializer(stddev=0.02)

    # These placeholders are used for input into the generator and discriminator, respectively.
    real_in = tf.placeholder(shape=[None, input_size, input_size, 3], dtype=tf.float32)  # Real images
    z_lat = tf.placeholder(shape=[None, z_size + number_continuous], dtype=tf.float32)  # Random vector

    # The below code is responsible for applying gradient descent to update
    # the GAN.
    trainerD = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
    trainerG = tf.train.AdamOptimizer(learning_rate=0.002, beta1=0.5)
    trainerQ = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)

    d_loss, g_loss, q_loss, Gz = tower_loss(real_in, z_lat)

    tvars = tf.trainable_variables()
    gen_vars = [v for v in tvars if v.name.startswith("generator/")]
    dis_vars = [v for v in tvars if v.name.startswith("discriminator/")]

    # tf.get_variable_scope().reuse_variables()  # important

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
        iter_ = infogan_faceutil.data_iterator(images, filenames, batch_size)
        for i in tqdm.tqdm(xrange(total_batch)):

            # Concatenate all c and z variables.
            zlat = latent_prior(z_size, batch_size)

            # Draw a sample batch from MNIST dataset.
            # start_time = time.time()
            image_flat, _ = iter_.next()
            # elapsed_time = time.time() - start_time
            image_batch = np.reshape(image_flat, [batch_size, 218, 178, 3])
            image_batch = center_crop(image_batch, crop_h=input_size)
            image_batch = (image_batch / 255.0 - 0.5) * 2.0

            # print "fetch data time: " + str(elapsed_time)

            _, dLoss = sess.run([update_D, d_loss], feed_dict={real_in: image_batch, z_lat: zlat})  # Update the discriminator
            # Update the generator, twice for good measure.
            _, gLoss = sess.run([update_G, g_loss], feed_dict={z_lat: zlat})
            _, qLoss = sess.run([update_Q, q_loss], feed_dict={z_lat: zlat})  # Update to optimize mutual information.

            if i % 50 == 0:
                print "epoch: " + str(epoch) + " Gen Loss: " + str(gLoss) + " Disc Loss: " + str(dLoss) + " Q Losses: " + str(qLoss)
                # Generate another z batch
                z_sample = np.random.uniform(-1.0, 1.0, size=[batch_size, z_size]).astype(np.float32)

                aa = np.reshape(np.array([[(ee / 4.5 - 1)] for ee in range(10) for tempj in range(10)]), [10, 10]).T
                bb = np.reshape(aa, [batch_size, 1])
                cc = np.zeros_like(bb)
                lcont_sample = np.hstack([bb, cc])

                # Concatenate all c and z variables.
                zlats = np.concatenate([z_sample, lcont_sample], 1).astype(np.float32)
                # Use new z to get sample images from generator.
                samples = sess.run(Gz, feed_dict={z_lat: zlats})
                if not os.path.exists(sample_directory):
                    os.makedirs(sample_directory)
                # Save sample generator images for viewing training
                # progress.
                infogan_faceutil.save_images(np.reshape(samples[0:batch_size], [batch_size, image_size, image_size, 3]), [10, 10], sample_directory + '/fig' + str(epoch) + str(i) + '.png')
        if epoch % 3 == 0:
            if not os.path.exists(model_directory):
                os.makedirs(model_directory)
            saver.save(sess, model_directory + '/model-epoch-' + str(epoch) + '.cptk', global_step=epoch)
            print "Saved Model"
        epoch += 1


def tower_loss(real_in, z_lat):

    real_in = tf.image.resize_images(real_in, [image_size, image_size])

    Gz = infogan_faceutil.generator(z_lat)  # Generates images from random z vectors
    # Produces probabilities for real images
    Dx, _, = infogan_faceutil.discriminator(real_in)
    # Produces probabilities for generator images
    Dg, QgCont1 = infogan_faceutil.discriminator(Gz, reuse=True)

    # These functions together define the optimization objective of the GAN.
    # This optimizes the discriminator.
    d_loss = -tf.reduce_mean(tf.log(Dx) + tf.log(1. - Dg), name='dloss')
    # g_loss = -tf.reduce_mean(tf.log((Dg / (1 - Dg))), name='gloss')  # KL Divergence optimizer
    g_loss = -tf.reduce_mean(tf.log(Dg), name='gloss')  # KL Divergence optimizer
    # d_loss = tf.Print(d_loss, [d_loss], 'd_loss = ', summarize=-1)

    # Combine losses for each of the continous variables.
    q_cont_loss1 = tf.reduce_sum(0.5 * tf.square(z_lat[:, z_size:] - QgCont1), reduction_indices=1)
    # q_cont_loss2 = tf.reduce_sum(0.5 * tf.square(z_lat[:, z_size + 4:] - QgCont2), reduction_indices=1)

    q_loss = tf.reduce_mean(q_cont_loss1, name='qloss')
    # q_cont_loss2 = tf.reduce_mean(q_cont_loss2)
    # q_loss = tf.add(q_cont_loss1, name='qloss')

    return d_loss, g_loss, q_loss, Gz


def latent_prior(z_size, batch_size):
    zs = np.random.uniform(-1.0, 1.0, size=[batch_size, z_size]).astype(np.float32)
    lcont1 = np.random.uniform(-1, 1, [batch_size, number_continuous])
    # lcont2 = np.random.uniform(-1, 1, [batch_size, number_continuous])
    # Concatenate all c and z variables.
    zlat = np.concatenate([zs, lcont1], 1).astype(np.float32)
    return zlat


if __name__ == '__main__':
    train_infogan()
    # test_infogan()
