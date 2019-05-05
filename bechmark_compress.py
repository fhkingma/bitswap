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

# seed
np.random.seed(100)

class ToInt:
    def __call__(self, pic):
        return pic * 255

def mnist():
    transform_ops = transforms.Compose([transforms.ToTensor(), ToInt()])
    mnist = datasets.MNIST(root="model/data/mnist", train=False, transform=transform_ops, download=True)
    return mnist.test_data.numpy()

def cifar():
    transform_ops = transforms.Compose([transforms.ToTensor(), ToInt()])
    cifar = datasets.CIFAR10(root="model/data/cifar", train=False, transform=transform_ops, download=True)
    return cifar.test_data

def imagenet():
    transform_ops = transforms.Compose([transforms.ToTensor(), ToInt()])
    imagenet = ImageNet(root='model/data/imagenet/test', file='test.npy', transform=transform_ops)
    if not os.path.exists("bitstreams/imagenet/indices"):
        randindices = np.random.choice(len(imagenet.dataset), size=(100, 100), replace=False)
        np.save("bitstreams/imagenet/indices", randindices)
    else:
        randindices = np.load("bitstreams/imagenet/indices")
    randindices = randindices.reshape(-1)
    return imagenet.dataset[randindices]

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

def bench_compressor(compress_fun, compressor_name, images, images_name):
    byts = compress_fun(images)
    n_bits = len(byts) * 8
    bitsperdim = n_bits / np.size(images)
    print(f"Dataset: {images_name}. Compressor: {compressor_name}. Rate: {bitsperdim:.2f} bits/dim.")

if __name__ == "__main__":
    # MNIST
    images = mnist()
    bench_compressor(gzip_compress, "gzip", images, 'MNIST')
    bench_compressor(bz2_compress, "bz2", images, 'MNIST')
    bench_compressor(lzma_compress, "lzma", images, 'MNIST')
    bench_compressor(
        pimg_compress("PNG", optimize=True), "PNG", images, 'MNIST')
    bench_compressor(
        pimg_compress('WebP', lossless=True, quality=100), "WebP", images, 'MNIST')
    print("")

    # CIFAR-10
    images = cifar()
    bench_compressor(gzip_compress, "gzip", images, 'CIFAR-10')
    bench_compressor(bz2_compress, "bz2", images, 'CIFAR-10')
    bench_compressor(lzma_compress, "lzma", images, 'CIFAR-10')
    bench_compressor(
        pimg_compress("PNG", optimize=True), "PNG", images, 'CIFAR-10')
    bench_compressor(
        pimg_compress('WebP', lossless=True, quality=100), "WebP", images, 'CIFAR-10')
    print("")

    # ImageNet
    images = imagenet()
    bench_compressor(gzip_compress, "gzip", images, 'ImageNet')
    bench_compressor(bz2_compress, "bz2", images, 'ImageNet')
    bench_compressor(lzma_compress, "lzma", images, 'ImageNet')
    bench_compressor(
        pimg_compress("PNG", optimize=True), "PNG", images, 'ImageNet')
    bench_compressor(
        pimg_compress('WebP', lossless=True, quality=100), "WebP", images, 'ImageNet')
