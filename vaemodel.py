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
        w_initializer = xavier_init(input_dims, output_dims)
        # w_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)
        weight = tf.get_variable('weight', shape, initializer=w_initializer)
        flat_x = tf.reshape(x, [batch, -1])
        fc = tf.matmul(tf.to_float(flat_x), weight)
        if with_bias:
            b_initializer = xavier_init(input_dims, output_dims)
            # b_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)
            bias = tf.get_variable('bias', [output_dims], initializer=b_initializer)
            fc = tf.nn.bias_add(fc, bias)
        if with_norm:
            fc = batch_norm(fc, output_dims, [0])
        return fc


def conv2d(x, input_filters, output_filters, kernel, strides, padding='SAME', with_bias=True):
    with tf.variable_scope('conv') as scope:

        shape = [kernel, kernel, input_filters, output_filters]
        w_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)
        weight = tf.get_variable('weight', shape, initializer=w_initializer)
        # weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        convolved = tf.nn.conv2d(x, weight, strides=[1, strides, strides, 1], padding=padding, name='conv')
        if with_bias:
            b_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)
            bias = tf.get_variable('bias', [output_filters], initializer=b_initializer)
            # bias = tf.Variable(tf.truncated_normal([output_filters], stddev=0.1), name='bias')
            convolved = tf.nn.bias_add(convolved, bias)
        normalized = batch_norm(convolved, output_filters)
        return normalized


def conv2d_transpose(x, input_filters, output_filters, kernel, strides, padding='SAME', with_bias=False):
    with tf.variable_scope('conv_transpose') as scope:

        shape = [kernel, kernel, output_filters, input_filters]
        w_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)
        weight = tf.get_variable('weight', shape, initializer=w_initializer)
        # weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')

        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1] * strides
        width = tf.shape(x)[2] * strides
        output_shape = tf.pack([batch_size, height, width, output_filters])
        convolved = tf.nn.conv2d_transpose(x, weight, output_shape, strides=[
                                           1, strides, strides, 1], padding=padding, name='conv_transpose')
        if with_bias:
            b_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)
            bias = tf.get_variable('bias', [output_filters], initializer=b_initializer)
            # bias = tf.Variable(tf.truncated_normal([output_filters], stddev=0.1), name='bias')
            convolved = tf.nn.bias_add(convolved, bias)
        normalized = batch_norm(convolved, output_filters)
        return normalized


def batch_norm(x, size, axes=[0, 1, 2]):
    batch_mean, batch_var = tf.nn.moments(x, axes, keep_dims=True)
    beta = tf.get_variable('beta', initializer=tf.zeros_initializer([size]))
    scale = tf.get_variable('scale', initializer=tf.ones_initializer([size]))
    epsilon = 1e-3
    return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon, name='batch')


def encoder(x, input_dims, latent_dims, reuse=False):
    with tf.variable_scope('encoder', reuse=reuse) as scope:
        with tf.variable_scope('en_fc_1'):
            en_fc_1 = tf.nn.softplus(fully_connect(x, input_dims, hidden))
        with tf.variable_scope('en_fc_2'):
            en_fc_2 = tf.nn.softplus(fully_connect(en_fc_1, hidden, hidden))
        with tf.variable_scope('en_fc_3_mean'):
            en_fc_3_mean = fully_connect(en_fc_2, hidden, latent_dims)
        with tf.variable_scope('en_fc_3_log_sigma'):
            en_fc_3_log_sigma = fully_connect(en_fc_2, hidden, latent_dims)
        return en_fc_3_mean, en_fc_3_log_sigma


def decoder(x, input_dims, output_dims, reuse=False):
    with tf.variable_scope('decoder', reuse=reuse) as scope:
        with tf.variable_scope('de_fc_1'):
            de_fc_1 = tf.nn.softplus(fully_connect(x, input_dims, hidden))
        with tf.variable_scope('de_fc_2'):
            de_fc_2 = tf.nn.softplus(fully_connect(de_fc_1, hidden, hidden))
        with tf.variable_scope('de_fc_3'):
            de_fc_3 = tf.nn.sigmoid(fully_connect(de_fc_2, hidden, output_dims))
        return de_fc_3


def encoder_decoder(x, input_dims, latent_dims, output_dims, reuse=False):
    with tf.variable_scope('encoder_decoder', reuse=reuse) as scope:
        # with tf.variable_scope('encoder'):
        e1, _ = encoder(x, input_dims, latent_dims)
        # with tf.variable_scope('decoder'):
        d1 = decoder(e1, latent_dims, output_dims)
        return d1, e1


def encoder_z_decoder(x, input_dims, latent_dims, output_dims, reuse=False):
    with tf.variable_scope('encoder_z_decoder', reuse=reuse) as scope:
        z_mean, z_log_sigma_sq = encoder(x, input_dims, latent_dims)
        eps = tf.random_normal((x.get_shape().as_list()[0], latent_dims), 0, 1, dtype=tf.float32)
        # eps_initializer = tf.random_normal_initializer(mean=0.0, stddev=1)
        # eps = tf.get_variable('eps', (x.get_shape().as_list()[0], latent_dims), initializer=eps_initializer)
        z = tf.add(z_mean, tf.mul(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))
        # z = tf.Print(z, [tf.reduce_sum(z)], 'z = ', summarize=20, first_n=7)
        d1 = decoder(z, latent_dims, output_dims)
        return d1, z_mean, z_log_sigma_sq


def discriminator(x, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse) as scope:
        shape = x.get_shape().as_list()
        input_dims = shape[1]
        with tf.variable_scope('dis_fc_1'):
            dis_fc_1 = tf.nn.relu(fully_connect(x, input_dims, hidden))
        with tf.variable_scope('dis_fc_2'):
            dis_fc_2 = tf.nn.relu(fully_connect(dis_fc_1, hidden, hidden))
        with tf.variable_scope('dis_fc_3'):
            dis_fc_3 = tf.nn.sigmoid(fully_connect(dis_fc_2, hidden, 1))
        return dis_fc_3


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
