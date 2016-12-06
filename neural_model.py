import tensorflow as tf

mean_pixel = [123.68, 116.779, 103.939]


def conv2d(x, input_filters, output_filters, kernel, strides, padding='SAME', with_bias=True):
    with tf.variable_scope('conv') as scope:

        shape = [kernel, kernel, input_filters, output_filters]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        convolved = tf.nn.conv2d(x, weight, strides=[1, strides, strides, 1], padding=padding, name='conv')
        if with_bias:
            bias = tf.Variable(tf.truncated_normal([output_filters], stddev=0.1), name='bias')
            convolved = tf.nn.bias_add(convolved, bias)
        normalized = instance_norm(convolved)

        return normalized


def instance_norm(x):
    epsilon = 1e-9
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    return tf.div(tf.sub(x, mean), tf.sqrt(tf.add(var, epsilon)))


def conv2d_transpose(x, input_filters, output_filters, kernel, strides, padding='SAME', with_bias=False):
    with tf.variable_scope('conv_transpose') as scope:

        shape = [kernel, kernel, output_filters, input_filters]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')

        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1] * strides
        width = tf.shape(x)[2] * strides
        output_shape = tf.pack([batch_size, height, width, output_filters])
        convolved = tf.nn.conv2d_transpose(x, weight, output_shape, strides=[
                                           1, strides, strides, 1], padding=padding, name='conv_transpose')
        if with_bias:
            bias = tf.Variable(tf.truncated_normal([output_filters], stddev=0.1), name='bias')
            convolved = tf.nn.bias_add(convolved, bias)
        normalized = instance_norm(convolved)
        return normalized


def batch_norm(x, size):
    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], keep_dims=True)
    beta = tf.Variable(tf.zeros([size]), name='beta')
    scale = tf.Variable(tf.ones([size]), name='scale')
    epsilon = 1e-3
    return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon, name='batch')


def residual(x, filters, kernel, strides, padding='SAME'):
    with tf.variable_scope('residual') as scope:
        conv1 = conv2d(x, filters, filters, kernel, strides, padding=padding)
        conv2 = conv2d(tf.nn.relu(conv1), filters, filters, kernel, strides, padding=padding)
        residual = x + conv2

        return residual


def gen_v4(x):
    with tf.variable_scope('gen_v4') as scope:
        with tf.variable_scope('conv1_1'):
            conv1_1 = tf.nn.relu(conv2d(x, 3, 48, 5, 1))
        with tf.variable_scope('conv2_1'):
            conv2_1 = tf.nn.relu(conv2d(conv1_1, 48, 32, 5, 1))
        with tf.variable_scope('conv3_1'):
            conv3_1 = tf.nn.relu(conv2d(conv2_1, 32, 64, 3, 1))
        with tf.variable_scope('conv4_1'):
            conv4_1 = tf.nn.relu(conv2d(conv3_1, 64, 32, 5, 1))
        with tf.variable_scope('conv5_1'):
            conv5_1 = tf.nn.relu(conv2d(conv4_1, 32, 48, 5, 1))
        with tf.variable_scope('conv6_1'):
            conv6_1 = tf.nn.relu(conv2d(conv5_1, 48, 32, 5, 1, with_bias=False))
        with tf.variable_scope('conv7_1'):
            conv7_1 = tf.nn.tanh(conv2d(conv6_1, 32, 3, 3, 1, with_bias=False))
        with tf.variable_scope('raw'):
            raw = conv7_1 * 127.5 + 127.5
        with tf.variable_scope('res1'):
            res1 = (raw - mean_pixel) * 0.4 + x * 0.6
        return res1


def gen_v3(x):
    with tf.variable_scope('gen_v3') as scope:
        with tf.variable_scope('conv1'):
            conv1 = tf.nn.relu(conv2d(x, 3, 64, 5, 2))
        with tf.variable_scope('conv1_1'):
            conv1_1 = tf.nn.relu(conv2d(conv1, 64, 48, 3, 1))
        with tf.variable_scope('conv2'):
            conv2 = tf.nn.relu(conv2d(conv1_1, 48, 128, 5, 2))
        with tf.variable_scope('conv2_1'):
            conv2_1 = tf.nn.relu(conv2d(conv2, 128, 96, 3, 1))
        with tf.variable_scope('conv3'):
            conv3 = tf.nn.relu(conv2d(conv2_1, 96, 256, 5, 2))
        with tf.variable_scope('conv3_1'):
            conv3_1 = tf.nn.relu(conv2d(conv3, 256, 192, 3, 1))
        with tf.variable_scope('deconv1'):
            deconv1 = tf.nn.relu(conv2d_transpose(conv3_1, 192, 128, 7, 2))
        with tf.variable_scope('res1'):
            res1 = deconv1 + conv2
        with tf.variable_scope('conv4_1'):
            conv4_1 = tf.nn.relu(conv2d(res1, 128, 160, 3, 1))
        with tf.variable_scope('deconv2'):
            deconv2 = tf.nn.relu(conv2d_transpose(conv4_1, 160, 64, 7, 2))
        with tf.variable_scope('res2'):
            res2 = deconv2 + conv1
        with tf.variable_scope('conv5_1'):
            conv5_1 = tf.nn.relu(conv2d(res2, 64, 96, 3, 1))
        with tf.variable_scope('deconv3'):
            deconv3 = tf.nn.tanh(conv2d_transpose(conv5_1, 96, 3, 8, 2))
        with tf.variable_scope('raw'):
            raw = deconv3 * 127.5 + 127.5
        with tf.variable_scope('res3'):
            res3 = (raw - mean_pixel) * 0.4 + x * 0.6
        return res3


def net(image):
    with tf.variable_scope('g0'):
        g0 = gen_v4(image)
    with tf.variable_scope('g1'):
        g1 = gen_v3(g0)
    with tf.variable_scope('g2'):
        g2 = gen_v3(g1)
    with tf.variable_scope('g3'):
        g3 = gen_v4(g2)

    return g0, g1, g2, g3
