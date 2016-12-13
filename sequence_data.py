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


def get_image(image_path, height, width, mode='RGB'):
    return scipy.misc.imresize(scipy.misc.imread(image_path, mode=mode), [height, width]).astype(np.float)


def get_dataset(path, dimh, dimw, channel=3):
    filenames = [join(path, f) for f in listdir(path) if isfile(join(path, f)) & f.lower().endswith('jpg')]
    filenum = len(filenames)
    size = dimh * dimw * channel
    seg = 1000
    chunknum = filenum / seg
    chunknum_tmp = chunknum
    remind = filenum % seg
    # make a dataset
    f = h5py.File('/home/jump/data/img_align_celeba.h5', 'w')
    images_h5py = f.create_dataset("images", shape=(chunknum, size), maxshape=(filenum, size), chunks=(chunknum, size), compression="gzip")
    filenames_h5py = f.create_dataset('filenames', data=filenames, compression="gzip")
    row_count = 0
    for jj in tqdm.tqdm(range(seg)):
        if jj == seg - 1:
            chunknum_tmp = chunknum + remind
        images_batch = np.zeros((chunknum_tmp, dimh * dimw * channel), dtype=np.uint8)
        for ii in range(chunknum_tmp):
            # for i in tqdm.tqdm(range(10)):
            image = get_image(filenames[ii + jj * chunknum], dimh, dimw)
            # images[i] = image.flatten()
            images_batch[ii] = image.flatten()
            # get the metadata
        images_h5py.resize(row_count + chunknum_tmp, axis=0)
        images_h5py[row_count:] = images_batch
        row_count += chunknum_tmp
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

    get_dataset('/home/jump/data/img_align_celeba', 218, 178, channel=3)
    '''
    with h5py.File(''.join(['datasets/coco_style-256.h5']), 'r') as hf:
        grams = hf['images'].value
        filenames = hf['filenames'].value
    '''
