import tensorflow as tf

mean_pixel = [123.68, 116.779, 103.939]


def fully_connect(x, input_dims, output_dims, with_bias=True, reuse=False):
    with tf.variable_scope('fully', reuse=reuse) as scope:
        batch = x.get_shape().as_list()[0]
        shape = [input_dims, output_dims]
        w_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)
        weight = tf.get_variable('weight', shape, initializer=w_initializer)
        # weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        flat_x = tf.reshape(x, [batch, -1])
        fc = tf.matmul(tf.to_float(flat_x), weight)
        if with_bias:
            b_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)
            bias = tf.get_variable('bias', [output_dims], initializer=b_initializer)
            # bias = tf.Variable(tf.truncated_normal([output_dims], stddev=0.1), name='bias')
            fc = tf.nn.bias_add(fc, bias)
        normalized = batch_norm(fc, output_dims, [0])
        return normalized


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


def residual(x, filters, kernel, strides, padding='SAME'):
    with tf.variable_scope('residual') as scope:
        conv1 = conv2d(x, filters, filters, kernel, strides, padding=padding)
        conv2 = conv2d(tf.nn.relu(conv1), filters, filters, kernel, strides, padding=padding)
        residual = x + conv2
        return residual


def encoder(x, input_dims, latent_dims, reuse=False):
    with tf.variable_scope('encoder', reuse=reuse) as scope:
        with tf.variable_scope('en_fc_1'):
            en_fc_1 = tf.nn.relu(fully_connect(x, input_dims, 1000))
        with tf.variable_scope('en_fc_2'):
            en_fc_2 = tf.nn.relu(fully_connect(en_fc_1, 1000, 1000))
        with tf.variable_scope('en_fc_3'):
            en_fc_3 = fully_connect(en_fc_2, 1000, latent_dims)
        return en_fc_3


def decoder(x, input_dims, output_dims, reuse=False):
    with tf.variable_scope('decoder', reuse=reuse) as scope:
        with tf.variable_scope('de_fc_1'):
            de_fc_1 = tf.nn.relu(fully_connect(x, input_dims, 1000))
        with tf.variable_scope('de_fc_2'):
            de_fc_2 = tf.nn.relu(fully_connect(de_fc_1, 1000, 1000))
        with tf.variable_scope('de_fc_3'):
            de_fc_3 = tf.nn.sigmoid(fully_connect(de_fc_2, 1000, output_dims)) * 256.0
        return de_fc_3


def encoder_decoder(x, input_dims, latent_dims, output_dims, reuse=False):
    with tf.variable_scope('encoder_decoder', reuse=reuse) as scope:    
        # with tf.variable_scope('encoder'):
        e1 = encoder(x, input_dims, latent_dims)
        # with tf.variable_scope('decoder'):
        d1 = decoder(e1, latent_dims, output_dims)
        return d1, e1


def discriminator(x, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse) as scope:
        shape = x.get_shape().as_list()
        input_dims = shape[1]
        with tf.variable_scope('dis_fc_1'):
            dis_fc_1 = tf.nn.relu(fully_connect(x, input_dims, 1000))
        with tf.variable_scope('dis_fc_2'):
            dis_fc_2 = tf.nn.relu(fully_connect(dis_fc_1, 1000, 1000))
        with tf.variable_scope('dis_fc_3'):
            dis_fc_3 = tf.nn.sigmoid(fully_connect(dis_fc_2, 1000, 1))
        return dis_fc_3


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


def discriminator2(x, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
            batch = x.get_shape().as_list()[0]
            shape = [2, 1000]
            W_initializer = tf.random_uniform_initializer(-1.0, 1.0)
            weight = tf.get_variable('weight', shape, initializer=W_initializer)
            # weight = tf.get_variable("weight", tf.truncated_normal(shape, stddev=0.1))
            flat_x = tf.reshape(x, [batch, -1])
            fc = tf.matmul(tf.to_float(flat_x), weight)
            dis_fc_1 = tf.nn.relu(fc)
            return dis_fc_1


def show_variables(variales):
    for i in range(len(variales)):
        print(variales[i].name)


def test():
    prior1 = get_prior([1] + [2])
    prior2 = get_prior([1] + [2])
    aaa = discriminator2(prior1)
    bbb = discriminator2(prior2, reuse=True)
    varrr = tf.all_variables()
    names = show_variables(varrr)
    print '1'


if __name__ == '__main__':
    test()
