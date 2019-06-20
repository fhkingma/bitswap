import io
import gzip
import bz2
import lzma
import numpy as np
from utils.torch.modules import ImageNet
import os

from torchvision import datasets, transforms
import PIL.Image as pimg

# code that applies benchmark compressors on the three datasets (MNIST, CIFAR-10 and ImageNet)
# heavily based on benchmark_compressors.py from https://github.com/bits-back/bits-back

# seed (for reproducibility MUST be the same for the whole project on the same machine)
np.random.seed(100)

# method to extract maximum amount of pixel-blocks of certain size from certain image
def extract_blocks(arr, block_size=(32, 32)):
    nrows, ncols = block_size
    h, w, c = arr.shape
    if h % nrows != 0:
        h -= h % nrows
        arr = arr[:h]
    if w % ncols != 0:
        w -= w % ncols
        arr = arr[:,:w]
    return (arr.reshape(h//nrows, nrows, -1, ncols, c)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols, c)), h, w

# method to reconstruct image from the extracted pixel-blocks
# note: this returns the original images being cropped to multiples of 32 pixels on each side
def unextract_blocks(arr, h, w):
    n, nrows, ncols, c = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols, c)
               .swapaxes(1,2)
               .reshape(h, w, c))
class ToInt:
    def __call__(self, pic):
        return pic * 255

def mnist(exp):
    transform_ops = transforms.Compose([transforms.ToTensor(), ToInt()])
    mnist = datasets.MNIST(root="model/data/mnist", train=False, transform=transform_ops, download=True)
    return mnist.test_data.numpy()[np.random.choice(len(mnist.test_data), size=(100, 100), replace=False)[exp]]

def cifar(exp):
    transform_ops = transforms.Compose([transforms.ToTensor(), ToInt()])
    cifar = datasets.CIFAR10(root="model/data/cifar", train=False, transform=transform_ops, download=True)
    return cifar.test_data[np.random.choice(len(cifar.test_data), size=(100, 100), replace=False)[exp]]

def imagenet(exp):
    transform_ops = transforms.Compose([transforms.ToTensor(), ToInt()])
    data = ImageNet(root='model/data/imagenet/test', file='test.npy', transform=transform_ops)
    if not os.path.exists("bitstreams/imagenet/indices"):
        randindices = np.random.choice(len(data.dataset), size=(100, 100), replace=False)
        np.save("bitstreams/imagenet/indices", randindices)
    else:
        randindices = np.load("bitstreams/imagenet/indices")
    return data.dataset[randindices[exp]]

def gzip_compress(images):
    images = np.packbits(images) if images.dtype is np.dtype(bool) else images
    assert images.dtype == np.dtype('uint8')
    return gzip.compress(images.tobytes())

def bz2_compress(images):
    images = np.packbits(images) if images.dtype is np.dtype(bool) else images
    assert images.dtype == np.dtype('uint8')
    return bz2.compress(images.tobytes())

def lzma_compress(images):
    images = np.packbits(images) if images.dtype is np.dtype(bool) else images
    assert images.dtype == np.dtype('uint8')
    return lzma.compress(images.tobytes())

def pimg_compress(format='PNG', **params):
    def compress_fun(images):
        compressed_data = bytearray()
        for n, image in enumerate(images):
            image = pimg.fromarray(image)
            img_bytes = io.BytesIO()
            image.save(img_bytes, format=format, **params)
            compressed_data.extend(img_bytes.getvalue())
        return compressed_data
    return compress_fun

def gz_and_pimg(images, format='PNG', **params):
    pimg_compressed_data = pimg_compress(images, format, **params)
    return gzip.compress(pimg_compressed_data)

def bench_compressor(compress_fun, images):
    byts = compress_fun(images)
    n_bits = len(byts) * 8
    bitsperdim = n_bits / np.size(images)
    return bitsperdim

if __name__ == "__main__":
    gzip_list = []
    bz2_list = []
    lzma_list = []
    png_list = []
    webp_list = []
    # MNIST
    print(f"Compressing MNIST test set")
    for exp in range(100):
        images = mnist(exp)
        gzip_list.append(bench_compressor(gzip_compress, images))
        bz2_list.append(bench_compressor(bz2_compress, images))
        lzma_list.append(bench_compressor(lzma_compress, images))
        png_list.append(bench_compressor(
            pimg_compress("PNG", optimize=True), images))
        webp_list.append(bench_compressor(
            pimg_compress('WebP', lossless=True, quality=100), images))
    print(f"gzip: {np.mean(gzip_list):.2f} bits/dim")
    print(f"bz2: {np.mean(bz2_list):.2f} bits/dim")
    print(f"lzma: {np.mean(lzma_list):.2f} bits/dim")
    print(f"png: {np.mean(png_list):.2f} bits/dim")
    print(f"webp: {np.mean(webp_list):.2f} bits/dim")
    print("")

    gzip_list = []
    bz2_list = []
    lzma_list = []
    png_list = []
    webp_list = []
    print(f"Compressing CIFAR-10 test set")
    for exp in range(100):
        # CIFAR-10
        images = cifar(exp)
        # MNIST
        gzip_list.append(bench_compressor(gzip_compress, images))
        bz2_list.append(bench_compressor(bz2_compress, images))
        lzma_list.append(bench_compressor(lzma_compress, images))
        png_list.append(bench_compressor(
            pimg_compress("PNG", optimize=True), images))
        webp_list.append(bench_compressor(
            pimg_compress('WebP', lossless=True, quality=100), images))
    print(f"gzip: {np.mean(gzip_list):.2f} bits/dim")
    print(f"bz2: {np.mean(bz2_list):.2f} bits/dim")
    print(f"lzma: {np.mean(lzma_list):.2f} bits/dim")
    print(f"png: {np.mean(png_list):.2f} bits/dim")
    print(f"webp: {np.mean(webp_list):.2f} bits/dim")
    print("")

    gzip_list = []
    bz2_list = []
    lzma_list = []
    png_list = []
    webp_list = []
    print(f"Compressing 10000 images from ImageNet test set")
    for exp in range(100):
        # ImageNet
        images = imagenet(exp)
        gzip_list.append(bench_compressor(gzip_compress, images))
        bz2_list.append(bench_compressor(bz2_compress, images))
        lzma_list.append(bench_compressor(lzma_compress, images))
        png_list.append(bench_compressor(
            pimg_compress("PNG", optimize=True), images))
        webp_list.append(bench_compressor(
            pimg_compress('WebP', lossless=True, quality=100), images))
    print(f"gzip: {np.mean(gzip_list):.2f} bits/dim")
    print(f"bz2: {np.mean(bz2_list):.2f} bits/dim")
    print(f"lzma: {np.mean(lzma_list):.2f} bits/dim")
    print(f"png: {np.mean(png_list):.2f} bits/dim")
    print(f"webp: {np.mean(webp_list):.2f} bits/dim")
    print("")
