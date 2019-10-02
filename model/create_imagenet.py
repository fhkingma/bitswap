import numpy as np
import scipy.ndimage
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

# function to convert ImageNet folder files into a Numpy compatible .npy file
# based on create_imagenet_benchmark_datasets.py from https://github.com/aravindsrinivas/flowpp
def convert_path_to_npy(*, path='~/train_32x32', outfile='~/train_32x32.npy'):
    assert isinstance(path, str), "Expected a string input for the path"
    assert os.path.exists(path), "Input path doesn't exist"

    # make list of files
    files = [f for f in listdir(path) if isfile(join(path, f))]
    print('Number of valid images is:', len(files))
    imgs = []

    # check all images for correct shapes etc. and dump them into
    for i in tqdm(range(len(files))):
        print(i)
        img = scipy.ndimage.imread(join(path, files[i]))
        img = img.astype('uint8')
        assert img.shape == (32, 32, 3)
        assert np.max(img) <= 255
        assert np.min(img) >= 0
        assert img.dtype == 'uint8'
        assert isinstance(img, np.ndarray)
        imgs.append(img)

    resolution_x, resolution_y = img.shape[0], img.shape[1]
    imgs = np.asarray(imgs).astype('uint8')
    assert imgs.shape[1:] == (resolution_x, resolution_y, 3)
    assert np.max(imgs) <= 255
    assert np.min(imgs) >= 0
    print('Total number of images is:', imgs.shape[0])
    print('All assertions done, dumping into npy file')
    os.mkdir(os.path.dirname(os.path.abspath(outfile)))
    np.save(outfile, imgs)

if __name__ == '__main__':
    convert_path_to_npy(path='data/train_32x32', outfile='data/imagenet/train/train.npy')
    convert_path_to_npy(path='data/valid_32x32', outfile='data/imagenet/test/test.npy')