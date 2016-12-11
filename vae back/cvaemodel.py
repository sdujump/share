import tensorflow as tf
import numpy as np

mean_pixel = [123.68, 116.779, 103.939]
hidden = 500


def xavier_init(input_dims, output_dims, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant * np.sqrt(6.0 / (input_dims + output_dims))
    high = constant * np.sqrt(6.0 / (input_dims + output_dims))
    # tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)
    return tf.random_uniform_initializer(minval=low, maxval=high)


def fully_connect(x, input_dims, output_dims, with_bias=True, with_norm=False, reuse=False):
    with tf.variable_scope('fully', reuse=reuse) as scope:
        batch = x.get_shape().as_list()[0]
        shape = [input_dims, output_dims]
        # w_initializer = xavier_init(input_dims, output_dims)
        w_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)
        weight = tf.get_variable('weight', shape, initializer=w_initializer)
        # weight = tf.Variable(tf.random_normal(shape, 0, 0.1, dtype=tf.float32), name='weight')
        flat_x = tf.reshape(x, [batch, -1])
        fc = tf.matmul(tf.to_float(flat_x), weight)
        if with_bias:
            # b_initializer = xavier_init(input_dims, output_dims)
            b_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)
            bias = tf.get_variable(
                'bias', [output_dims], initializer=b_initializer)
            # bias = tf.Variable(tf.random_normal([output_dims], 0, 0.1, dtype=tf.float32), name='bias')
            fc = tf.nn.bias_add(fc, bias)
        if with_norm:
            fc = batch_norm(fc, output_dims, [0])
        return fc


def conv2d(x, input_filters, output_filters, kernel, strides, activate=tf.nn.relu, padding='SAME', with_bias=True, with_norm=False, reuse=False):
    with tf.variable_scope('conv', reuse=reuse) as scope:

        shape = [kernel, kernel, input_filters, output_filters]
        w_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)
        weight = tf.get_variable('weight', shape, initializer=w_initializer)
        # weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        convolved = tf.nn.conv2d(
            x, weight, strides=[1, strides, strides, 1], padding=padding, name='conv')
        if with_bias:
            b_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)
            bias = tf.get_variable(
                'bias', [output_filters], initializer=b_initializer)
            # bias = tf.Variable(tf.truncated_normal([output_filters], stddev=0.1), name='bias')
            convolved = tf.nn.bias_add(convolved, bias)
        if with_norm:
            convolved = batch_norm(convolved, output_filters)
        convolved = activate(convolved)
        return convolved


def conv2d_transpose(x, input_filters, output_filters, kernel, strides, activate=tf.nn.relu, padding='SAME', with_bias=True, with_norm=False, reuse=False):
    with tf.variable_scope('conv_transpose', reuse=reuse) as scope:

        shape = [kernel, kernel, output_filters, input_filters]
        w_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)
        weight = tf.get_variable('weight', shape, initializer=w_initializer)
        # weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')

        batch_size = x.get_shape().as_list()[0]
        height = x.get_shape().as_list()[1] * strides
        width = x.get_shape().as_list()[2] * strides
        output_shape = tf.pack([batch_size, height, width, output_filters])
        convolved = tf.nn.conv2d_transpose(x, weight, output_shape, strides=[
                                           1, strides, strides, 1], padding=padding, name='conv_transpose')
        if with_bias:
            b_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)
            bias = tf.get_variable(
                'bias', [output_filters], initializer=b_initializer)
            # bias = tf.Variable(tf.truncated_normal([output_filters], stddev=0.1), name='bias')
            convolved = tf.nn.bias_add(convolved, bias)
        if with_norm:
            convolved = batch_norm(convolved, output_filters)
        convolved = activate(convolved)
        return convolved


def batch_norm(x, size, axes=[0, 1, 2]):
    batch_mean, batch_var = tf.nn.moments(x, axes, keep_dims=True)
    beta = tf.get_variable('beta', initializer=tf.zeros_initializer([size]))
    scale = tf.get_variable('scale', initializer=tf.ones_initializer([size]))
    epsilon = 1e-3
    return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon, name='batch')


def conv_encoder(x, latent_dims, reuse=False):
        with tf.variable_scope('en_conv_1'):
            en_conv_1 = conv2d(x, 1, 16, 3, 2)
        with tf.variable_scope('en_conv_2'):
            en_conv_2 = conv2d(en_conv_1, 16, 32, 3, 1)
        with tf.variable_scope('en_conv_3'):
            en_conv_3 = conv2d(en_conv_2, 32, 32, 3, 1)
        with tf.variable_scope('en_conv_4'):
            en_conv_4 = conv2d(en_conv_3, 32, 16, 3, 2)
        with tf.variable_scope('en_fc_mean'):
            shape = en_conv_4.get_shape().as_list()
            size = shape[1] * shape[2] * shape[3]
            en_fc_mean = fully_connect(en_conv_4, size, latent_dims)
        with tf.variable_scope('en_fc_log_sigma'):
            en_fc_log_sigma = fully_connect(en_conv_4, size, latent_dims)
        return en_fc_mean, en_fc_log_sigma, shape


def conv_decoder(latnet, latent_dims, shape, reuse=False):
        with tf.variable_scope('de_fc'):
            size = shape[1] * shape[2] * shape[3]
            de_fc = tf.reshape(fully_connect(latnet, latent_dims, size), shape)
        with tf.variable_scope('de_conv_1'):
            de_conv_1 = conv2d_transpose(de_fc, shape[3], 32, 3, 2)
        with tf.variable_scope('de_conv_2'):
            de_conv_2 = conv2d_transpose(de_conv_1, 32, 32, 3, 1)
        with tf.variable_scope('de_conv_3'):
            de_conv_3 = conv2d_transpose(de_conv_2, 32, 16, 3, 1)
        with tf.variable_scope('de_conv_4'):
            de_conv_4 = conv2d_transpose(
                de_conv_3, 16, 1, 3, 2, activate=tf.nn.sigmoid)
        return de_conv_4


def conv_discriminator(x, reuse=False):
    with tf.variable_scope('dis_conv_1'):
        dis_conv_1 = conv2d(x, 1, 16, 3, 2)
    with tf.variable_scope('dis_conv_2'):
        dis_conv_2 = conv2d(dis_conv_1, 16, 32, 3, 1)
    with tf.variable_scope('dis_conv_3'):
        dis_conv_3 = conv2d(dis_conv_2, 32, 32, 3, 1)
    with tf.variable_scope('dis_conv_4'):
        dis_conv_4 = conv2d(dis_conv_3, 32, 16, 3, 2)
    with tf.variable_scope('dis_fc_latent'):
        shape = dis_conv_4.get_shape().as_list()
        size = shape[1] * shape[2] * shape[3]
        dis_fc_latent = tf.nn.relu(fully_connect(dis_conv_4, size, 100))
    with tf.variable_scope('dis_fc'):
        dis_fc = tf.nn.sigmoid(fully_connect(dis_fc_latent, 100, 1))
    return dis_fc, dis_fc_latent


def conv_encoder_z_decoder(x, latent_dims, reuse=False):
    with tf.variable_scope('encoder_z_decoder', reuse=reuse) as scope:
        with tf.variable_scope('encoder'):
            z_mean, z_log_sigma_sq, en_weight = conv_encoder(x, latent_dims)
            eps = tf.Variable(tf.random_normal((x.get_shape().as_list()[0], latent_dims), 0, 1, dtype=tf.float32), name='eps', trainable=False)
            # eps_initializer = tf.random_normal_initializer(mean=0.0, stddev=1)
            # eps = tf.get_variable('eps', (x.get_shape().as_list()[0], latent_dims), initializer=eps_initializer)
            z = tf.add(z_mean, tf.mul(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))
            # z = tf.Print(z, [tf.reduce_sum(z)], 'z = ', summarize=20, first_n=7)
        with tf.variable_scope('decoder'):
            d1 = conv_decoder(z, latent_dims, en_weight)
        return d1, z_mean, z_log_sigma_sq, z


def inference(x, latent_dims):
    with tf. variable_scope('inference') as scope:
        with tf.variable_scope('encoder'):
            z_mean, z_log_sigma_sq, en_weight = conv_encoder(x, latent_dims)
            eps = tf.Variable(tf.random_normal((x.get_shape().as_list()[0], latent_dims), 0, 1, dtype=tf.float32), name='eps', trainable=False)
            z = tf.add(z_mean, tf.mul(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))
        with tf.variable_scope('decoder'):
            reconst = conv_decoder(z, latent_dims, en_weight)
        with tf.variable_scope('decoder', reuse=True):
            pz = tf.Variable(tf.random_normal((x.get_shape().as_list()[0], latent_dims), 0, 1, dtype=tf.float32), name='pz', trainable=False)
            preconst = conv_decoder(pz, latent_dims, en_weight, reuse=True)
            # reconst, z_mean, z_log_sigma_sq, z = conv_encoder_z_decoder(x, latent_dims)
        with tf.variable_scope('discriminator'):
            dis_generate, _ = conv_discriminator(reconst)
        with tf.variable_scope('discriminator', reuse=True):
            dis_input, _ = conv_discriminator(x, reuse=True)
        with tf.variable_scope('discriminator', reuse=True):
            dis_p, _ = conv_discriminator(preconst, reuse=True)
        return reconst, z_mean, z_log_sigma_sq, dis_generate, dis_input, dis_p


def show_variables(variales):
    for i in range(len(variales)):
        print(variales[i].name)


'''
def get_prior(output_dims, type='uniform', reuse=False):
    with tf.variable_scope('prior', reuse=reuse) as scope:
        if type == 'uniform':
            p_initializer = tf.random_uniform_initializer(minval=-2, maxval=2)
            prior = tf.get_variable('prior', output_dims, initializer=p_initializer)
            # prior = tf.random_uniform(output_dims, -2, 2)
        else:
            p_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)
            prior = tf.get_variable('prior', output_dims, initializer=p_initializer)
            # prior = tf.random_normal(output_dims)
    return prior
'''
