import h5py  # for reading our dataset
from os import listdir
from os.path import isfile, join
import numpy as np
import tqdm  # making loops prettier
import scipy.misc
import os
from tensorflow.python.client import device_lib


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print device_lib.list_local_devices()


def get_image(image_path, height, width, mode='RGB'):
    return scipy.misc.imresize(scipy.misc.imread(image_path, mode=mode), [height, width]).astype(np.float)


def get_dataset(path, dimh, dimw, channel=3):
    filenames = [join(path, f) for f in listdir(path) if isfile(join(path, f)) & f.lower().endswith('jpg')]
    images = []
    # np.zeros((len(filenames), dim * dim * channel), dtype=np.uint8)
    # make a dataset
    for i in tqdm.tqdm(range(len(filenames))):
        # for i in tqdm.tqdm(range(10)):
        image = get_image(filenames[i], dimh, dimw)
        image = image.flatten()
        images.append(image)
        # get the metadata
    with h5py.File(''.join(['datasets/img_align_celeba.h5']), 'w') as f:
        images = f.create_dataset("images", data=images)
        filenames = f.create_dataset('filenames', data=filenames)
    print("dataset loaded")


def data_iterator(images, filenames, batch_size):
    """ A simple data iterator """
    batch_idx = 0
    while True:
        idxs = np.arange(0, len(images))
        np.random.shuffle(idxs)
        for batch_idx in range(0, len(images), batch_size):
            cur_idxs = idxs[batch_idx:batch_idx + batch_size]
            images_batch = images[cur_idxs]
            # images_batch = images_batch.astype("float32")
            names_batch = filenames[cur_idxs]
            yield images_batch, names_batch


if __name__ == '__main__':
    # tf.app.run()
    get_dataset('/home/jump/data/img_align_celeba', 218, 178, channel=3)
