import h5py  # for reading our dataset
from os import listdir
from os.path import isfile, join
import numpy as np
import tqdm  # making loops prettier
import scipy.misc
import os
from tensorflow.python.client import device_lib


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print device_lib.list_local_devices()


def get_image(image_path, width, height, mode='RGB'):
    return scipy.misc.imresize(scipy.misc.imread(image_path, mode=mode), [height, width]).astype(np.float)


def get_dataset(path, dim, channel=3):
    filenames = [join(path, f) for f in listdir(path) if isfile(join(path, f)) & f.lower().endswith('png')]
    filenum = len(filenames)
    size = dim * dim * channel
    seg = 100
    chunknum = filenum / seg
    chunknum_tmp = chunknum
    remind = filenum % seg
    # make a dataset
    f = h5py.File('datasets/coco_style-256.h5', 'w')
    images_h5py = f.create_dataset("images", shape=(chunknum, size), maxshape=(filenum, size), chunks=(chunknum, size))
    filenames_h5py = f.create_dataset('filenames', data=filenames)
    for jj in tqdm.tqdm(range(seg)):
        images_batch = []
        if jj == seg - 1:
            chunknum_tmp = chunknum_tmp + remind
        for ii in range(chunknum_tmp):
            # for i in tqdm.tqdm(range(10)):
            image = get_image(filenames[ii + jj * chunknum], dim, dim)
            # images[i] = image.flatten()
            images_batch.append(image.flatten())
            # get the metadata
        images_h5py[jj * chunknum:jj * chunknum + chunknum_tmp] = images_batch
        # filenames = f.create_dataset('filenames', data=filenames)
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

    get_dataset('coco_style', 256, channel=3)
    '''
    with h5py.File(''.join(['datasets/coco_style-256.h5']), 'r') as hf:
        grams = hf['images'].value
        filenames = hf['filenames'].value
    '''
