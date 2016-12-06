from scipy import misc
import os
import time
import tensorflow as tf
import vgg
import mxmodel2
from os import listdir
from os.path import isfile, join
from tensorflow.python.client import device_lib

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print device_lib.list_local_devices()

tf.app.flags.DEFINE_integer("CONTENT_WEIGHT", 5e0, "Weight for content features loss")
tf.app.flags.DEFINE_integer("STYLE_WEIGHT", 1e2, "Weight for style features loss")
tf.app.flags.DEFINE_integer("TV_WEIGHT", 1e-5, "Weight for total variation loss")
tf.app.flags.DEFINE_string("VGG_PATH", "imagenet-vgg-verydeep-19.mat", "Path to vgg model weights")
tf.app.flags.DEFINE_string("MODEL_PATH", "models", "Path to read/write trained models")
tf.app.flags.DEFINE_string("TRAIN_IMAGES_PATH", "contents", "Path to training images")
tf.app.flags.DEFINE_string("TRAIN_STYLES_PATH", "styles", "Path to style images")
tf.app.flags.DEFINE_string("CONTENT_LAYERS", "relu4_2", "Which VGG layer to extract content loss from")
tf.app.flags.DEFINE_string("STYLE_LAYERS", "relu1_2,relu2_2,relu3_2,relu4_2", "Which layers to extract style from")
tf.app.flags.DEFINE_string("SUMMARY_PATH", "tensorboard", "Path to store Tensorboard summaries")
tf.app.flags.DEFINE_string("STYLE_IMAGES", "style.png", "Styles to train")
tf.app.flags.DEFINE_float("STYLE_SCALE", 1.0, "Scale styles. Higher extracts smaller features")
tf.app.flags.DEFINE_string("CONTENT_IMAGES_PATH", None, "Path to content image(s)")
tf.app.flags.DEFINE_integer("IMAGE_SIZE", 256, "Size of output image")
tf.app.flags.DEFINE_integer("STYLE_SIZE", 512, "Size of output image")
tf.app.flags.DEFINE_integer("BATCH_SIZE", 1, "Number of concurrent images to train on")

FLAGS = tf.app.flags.FLAGS


def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0, 0, 0, 0], tf.pack(
        [-1, height - 1, -1, -1])) - tf.slice(layer, [0, 1, 0, 0], [-1, -1, -1, -1])
    x = tf.slice(layer, [0, 0, 0, 0], tf.pack(
        [-1, -1, width - 1, -1])) - tf.slice(layer, [0, 0, 1, 0], [-1, -1, -1, -1])
    return tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))

# TODO: Figure out grams and batch sizes! Doesn't make sense ..


def gram(layer):
    shape = tf.shape(layer)
    num_images = shape[0]
    num_filters = shape[3]
    size = tf.size(layer)
    filters = tf.reshape(layer, tf.pack([num_images, -1, num_filters]))
    grams = tf.batch_matmul(filters, filters, adj_x=True) / tf.to_float(size)  # / FLAGS.BATCH_SIZE)

    return grams


def get_image(path, size, epochs, shuffle=False, crop=True):
    filenames = [join(path, f) for f in listdir(path) if isfile(join(path, f)) & f.lower().endswith('jpg')]
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=epochs, shuffle=shuffle)
    reader = tf.WholeFileReader()
    _, img_bytes = reader.read(filename_queue)
    image = tf.image.decode_jpeg(img_bytes, channels=3)
    shape = tf.shape(image)
    if crop:
        h_ratio = size[0] / tf.to_float(shape[0])
        w_ratio = size[1] / tf.to_float(shape[1])
        ratio = tf.maximum(h_ratio, w_ratio)
        height = tf.ceil(tf.to_float(shape[0]) * ratio)
        weight = tf.ceil(tf.to_float(shape[1]) * ratio)
        before_crop = tf.image.resize_images(image, tf.to_int32([height] + [weight]))
        resized_image = tf.image.resize_image_with_crop_or_pad(before_crop, size[0], size[1])
    else:
        resized_image = tf.image.resize_images(image, size)
    mean_pixel = [123.68, 116.779, 103.939]
    processed_image = tf.to_float(resized_image) - mean_pixel
    return processed_image


def main(argv=None):

    if not os.path.exists(FLAGS.MODEL_PATH):
        os.makedirs(FLAGS.MODEL_PATH)
    mean_pixel = [123.68, 116.779, 103.939]
    # style_path = FLAGS.STYLE_IMAGES.split(',')
    style_paths = FLAGS.TRAIN_STYLES_PATH
    content_paths = FLAGS.TRAIN_IMAGES_PATH
    style_layers = FLAGS.STYLE_LAYERS.split(',')
    content_layers = FLAGS.CONTENT_LAYERS.split(',')

    # style_features_t = get_style_features(style_path, style_layers)

    # images = reader.image(FLAGS.BATCH_SIZE, FLAGS.IMAGE_SIZE, FLAGS.TRAIN_IMAGES_PATH)
    content_size = FLAGS.IMAGE_SIZE
    content_image = get_image(content_paths, [content_size] + [content_size], 2, False)
    content_images = tf.train.batch([content_image], 1, dynamic_pad=True)
    style_size = FLAGS.STYLE_SIZE
    style_image = get_image(style_paths, [style_size] + [style_size], None, False, crop=False)
    style_images = tf.train.batch([style_image], 1, dynamic_pad=True)

    generated = mxmodel2.net(content_images)

    total_loss = 0
    total_content = 0
    total_style = 0
    total_variation = 0

    style_net, _ = vgg.net(FLAGS.VGG_PATH, style_images)
    content_net, _ = vgg.net(FLAGS.VGG_PATH, content_images)

    for i in range(len(generated)):
        generated_net, _ = vgg.net(FLAGS.VGG_PATH, generated[i])

        content_loss = 0
        for layer in content_layers:
            content_vgg = content_net[layer]
            generated_vgg = generated_net[layer]
            size = tf.size(generated_vgg)
            content_loss += tf.nn.l2_loss(generated_vgg - content_vgg) / tf.to_float(size)
        content_loss = content_loss / len(content_layers)

        style_loss = 0
        for layer in style_layers:
            generated_vgg = generated_net[layer]
            style_vgg = style_net[layer]
            size = tf.square(tf.shape(style_vgg)[3])
            # for style_batch in style_gram:
            style_loss += tf.nn.l2_loss(tf.reduce_sum(gram(generated_vgg) - gram(style_vgg), 0)) / tf.to_float(size)
        style_loss = style_loss / len(style_layers)

        total_content += content_loss
        total_style += style_loss
        total_variation += total_variation_loss(generated[i])
        tf.image_summary('generated {}'.format(i), tf.saturate_cast(generated[i] + mean_pixel, tf.uint8))

    total_loss = FLAGS.STYLE_WEIGHT * total_style + FLAGS.CONTENT_WEIGHT * total_content + FLAGS.TV_WEIGHT * total_variation
    # loss = FLAGS.STYLE_WEIGHT * style_loss + FLAGS.CONTENT_WEIGHT * content_loss + FLAGS.TV_WEIGHT * total_variation_loss(generated)

    global_step = tf.Variable(0, name="global_step", trainable=False)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(total_loss, global_step=global_step)

    output_format = tf.saturate_cast(tf.concat(0, [generated[-1], content_images]) + mean_pixel, tf.uint8)

    tf.image_summary('content_images', tf.expand_dims(output_format[1], 0))
    tf.scalar_summary('total_content', total_content)
    tf.scalar_summary('total_style', total_style)
    tf.scalar_summary('total_variation', total_variation)
    tf.scalar_summary('total_loss', total_loss)
    summary_op = tf.merge_all_summaries()

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.all_variables())
        file = tf.train.latest_checkpoint(FLAGS.MODEL_PATH)
        if file:
            print('Restoring model from {}'.format(file))
            saver.restore(sess, file)
        else:
            print('New model initilized')
            sess.run(tf.initialize_all_variables())
            sess.run(tf.initialize_local_variables())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        summary_writer = tf.train.SummaryWriter(FLAGS.MODEL_PATH, sess.graph)
        start_time = time.time()
        try:
            while not coord.should_stop():
                _, loss_t, step = sess.run([train_op, total_loss, global_step])
                elapsed_time = time.time() - start_time
                start_time = time.time()
                if step % 10 == 0:
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)
                if step % 50 == 0:
                    print(step, loss_t, elapsed_time)
                    output_t = sess.run(output_format)
                    for i, raw_image in enumerate(output_t):
                        misc.imsave('test/out%s-%s.png' % (step, i + 1), raw_image)
                if step % 1000 == 0:
                    saver.save(sess, FLAGS.MODEL_PATH +
                               '/fast-style-model', global_step=step)
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
