from utils.torch.rand import *
from utils.torch.modules import ImageNet
from model.imagenetcrop_train import Model
from torch.utils.data import *
from discretization import *
from benchmark_compress import *
from torchvision import datasets, transforms
import random
import time
import argparse
from tqdm import tqdm
import pickle
import os
import scipy.ndimage
import os
from os import listdir
from os.path import isfile, join
import sys
from PIL import Image

class ANS:
    def __init__(self, pmfs, bits=31, quantbits=8):
        self.device = pmfs.device
        self.bits = bits
        self.quantbits = quantbits

        # mask of 2**bits - 1 bits
        self.mask = (1 << bits) - 1

        # normalization constants
        self.lbound = 1 << 32
        self.tail_bits = (1 << 32) - 1

        self.seq_len, self.support = pmfs.shape

        # compute pmf's and cdf's scaled up by 2**n
        multiplier = (1 << self.bits) - (1 << self.quantbits)
        self.pmfs = (pmfs * multiplier).long()

        # add ones to counter zero probabilities
        self.pmfs += torch.ones_like(self.pmfs)

        # add remnant to the maximum value of the probabilites
        self.pmfs[torch.arange(0, self.seq_len),torch.argmax(self.pmfs, dim=1)] += ((1 << self.bits) - self.pmfs.sum(1))

        # compute cdf's
        self.cdfs = torch.cumsum(self.pmfs, dim=1) # compute CDF (scaled up to 2**n)
        self.cdfs = torch.cat([torch.zeros([self.cdfs.shape[0], 1], dtype=torch.long, device=self.device), self.cdfs], dim=1) # pad with 0 at the beginning

        # move cdf's and pmf's the cpu for faster encoding and decoding
        self.cdfs = self.cdfs.cpu().numpy()
        self.pmfs = self.pmfs.cpu().numpy()

        assert self.cdfs.shape == (self.seq_len, self.support + 1)
        assert np.all(self.cdfs[:,-1] == (1 << bits))

    def encode(self, x, symbols):
        for i, s in enumerate(symbols):
            pmf = int(self.pmfs[i,s])
            if x[-1] >= ((self.lbound >> self.bits) << 32) * pmf:
                x.append(x[-1] >> 32)
                x[-2] = x[-2] & self.tail_bits
            x[-1] = ((x[-1] // pmf) << self.bits) + (x[-1] % pmf) + int(self.cdfs[i, s])
        return x

    def decode(self, x):
        sequence = np.zeros((self.seq_len,), dtype=np.int64)
        for i in reversed(range(self.seq_len)):
            masked_x = x[-1] & self.mask
            s = np.searchsorted(self.cdfs[i,:-1], masked_x, 'right') - 1
            sequence[i] = s
            x[-1] = int(self.pmfs[i,s]) * (x[-1] >> self.bits) + masked_x - int(self.cdfs[i, s])
            if x[-1] < self.lbound:
                x[-1] = (x[-1] << 32) | x.pop(-2)
        sequence = torch.from_numpy(sequence).to(self.device)
        return x, sequence

def compress(quantbits, nz, bitswap, gpu, blocks):
    # model and compression params
    zdim = 8*16*16
    zrange = torch.arange(zdim)
    xdim = 32**2 * 3
    xrange = torch.arange(xdim)
    ansbits = 31 # ANS precision
    type = torch.float64 # datatype throughout compression
    device = f"cuda:{gpu}" # gpu

    # set up the different channel dimension
    reswidth = 256

    # seed for replicating experiment and stability
    np.random.seed(100)
    random.seed(50)
    torch.manual_seed(50)
    torch.cuda.manual_seed(50)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # <=== MODEL ===>
    model = Model(xs = (3, 32, 32), nz=nz, zchannels=8, nprocessing=4, kernel_size=3, resdepth=8, reswidth=reswidth).to(device)
    model.load_state_dict(
        torch.load(f'model/params/imagenetcrop/nz4',
                   map_location=lambda storage, location: storage
                   )
    )
    model.eval()

    # get discretization bins for latent variables
    zendpoints, zcentres = discretize(nz, quantbits, type, device, model, "imagenetcrop")

    # get discretization bins for discretized logistic
    xbins = ImageBins(type, device, xdim)
    xendpoints = xbins.endpoints()
    xcentres = xbins.centres()

    # compression experiment params
    nblocks = blocks.shape[0]

    # < ===== COMPRESSION ===>
    # initialize compression
    model.compress()
    state = list(map(int, np.random.randint(low=1 << 16, high=(1 << 32) - 1, size=10000, dtype=np.uint32))) # fill state list with 'random' bits
    state[-1] = state[-1] << 32
    restbits = None

    # <===== SENDER =====>
    iterator = tqdm(range(nblocks), desc="Compression")
    for xi in iterator:
        x = blocks[xi]
        x = torch.from_numpy(x).to(device).view(xdim)

        if bitswap:
            # < ===== Bit-Swap ====>
            # inference and generative model
            for zi in range(nz):
                # inference model
                input = zcentres[zi - 1, zrange, zsym] if zi > 0 else xcentres[xrange, x.long()]
                mu, scale = model.infer(zi)(given=input)
                cdfs = logistic_cdf(zendpoints[zi].t(), mu, scale).t()# most expensive calculation?
                pmfs = cdfs[:, 1:] - cdfs[:, :-1]
                pmfs = torch.cat((cdfs[:,0].unsqueeze(1), pmfs, 1. - cdfs[:,-1].unsqueeze(1)), dim=1)

                # decode z
                state, zsymtop = ANS(pmfs, bits=ansbits, quantbits=quantbits).decode(state)

                # save excess bits for calculations
                if xi == zi == 0:
                    restbits = state.copy()
                    assert len(restbits) > 1, "too few initial bits" # otherwise initial state consists of too few bits

                # generative model
                z = zcentres[zi, zrange, zsymtop]
                mu, scale = model.generate(zi)(given=z)
                cdfs = logistic_cdf((zendpoints[zi - 1] if zi > 0 else xendpoints).t(), mu, scale).t() # most expensive calculation?
                pmfs = cdfs[:, 1:] - cdfs[:, :-1]
                pmfs = torch.cat((cdfs[:,0].unsqueeze(1), pmfs, 1. - cdfs[:,-1].unsqueeze(1)), dim=1)

                # encode z or x
                state = ANS(pmfs, bits=ansbits, quantbits=(quantbits if zi > 0 else 8)).encode(state, zsym if zi > 0 else x.long())

                zsym = zsymtop
        else:
            # < ===== BB-ANS ====>
            # inference and generative model
            zs = []
            for zi in range(nz):
                # inference model
                input = zcentres[zi - 1, zrange, zsym] if zi > 0 else xcentres[xrange, x.long()]
                mu, scale = model.infer(zi)(given=input)
                cdfs = logistic_cdf(zendpoints[zi].t(), mu, scale).t() # most expensive calculation?
                pmfs = cdfs[:, 1:] - cdfs[:, :-1]
                pmfs = torch.cat((cdfs[:, 0].unsqueeze(1), pmfs, 1. - cdfs[:, -1].unsqueeze(1)), dim=1)

                # decode z
                state, zsymtop = ANS(pmfs, bits=ansbits, quantbits=quantbits).decode(state)
                zs.append(zsymtop)

                zsym = zsymtop

            # save excess bits for calculations
            if xi == 0:
                restbits = state.copy()
                assert len(restbits) > 1  # otherwise initial state consists of too few bits

            for zi in range(nz):
                # generative model
                zsymtop = zs.pop(0)
                z = zcentres[zi, zrange, zsymtop]
                mu, scale = model.generate(zi)(given=z)
                cdfs = logistic_cdf((zendpoints[zi - 1] if zi > 0 else xendpoints).t(), mu, scale).t() # most expensive calculation?
                pmfs = cdfs[:, 1:] - cdfs[:, :-1]
                pmfs = torch.cat((cdfs[:, 0].unsqueeze(1), pmfs, 1. - cdfs[:, -1].unsqueeze(1)), dim=1)

                # encode z or x
                state = ANS(pmfs, bits=ansbits, quantbits=(quantbits if zi > 0 else 8)).encode(state, zsym if zi > 0 else x.long())

                zsym = zsymtop

            assert zs == []

        # prior
        cdfs = logistic_cdf(zendpoints[-1].t(), torch.zeros(1, device=device, dtype=type), torch.ones(1, device=device, dtype=type)).t()
        pmfs = cdfs[:, 1:] - cdfs[:, :-1]
        pmfs = torch.cat((cdfs[:, 0].unsqueeze(1), pmfs, 1. - cdfs[:, -1].unsqueeze(1)), dim=1)

        # encode prior
        state = ANS(pmfs, bits=ansbits, quantbits=quantbits).encode(state, zsymtop)

        # calculating bits
        totalbits = (len(state) - (len(restbits) - 1)) * 32

    bitsperdim = totalbits / (nblocks * 32 * 32 * 3)
    return bitsperdim

def convert_image_to_numpy(*, path=''):
    assert isinstance(path, str), "Expected a string input for the image path"
    assert os.path.exists(path), "Image path doesn't exist"
    assert isfile(path)
    imgs = []
    img = scipy.ndimage.imread(path)
    img = img.astype('uint8')
    img_valid = (img.shape[-1] == 3)
    if img_valid:
        old_h, old_w, _ = img.shape
        assert np.max(img) <= 255
        assert np.min(img) >= 0
        assert img.dtype == 'uint8'
        assert isinstance(img, np.ndarray)
        img, h, w = extract_blocks(img, block_size=(32,32))
        for k in range(img.shape[0]):
            imgs.append(img[k])
        resolution_x, resolution_y = img.shape[1], img.shape[2]
        imgs = np.asarray(imgs).astype('uint8')
        assert imgs.shape[1:] == (resolution_x, resolution_y, 3)
        assert np.max(imgs) <= 255
        assert np.min(imgs) >= 0
        if old_h != h and old_w != w:
            print(f'Image reshaped from ({old_h}, {old_w}, 3) to ({h}, {w}, 3)')
        else:
            print(f'Image shape is ({h}, {w}, 3)')
        return np.asarray(imgs), h, w, img_valid
    else:
        return None, None, None, img_valid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int) # assign to gpu
    args = parser.parse_args()
    print(args)
    gpu = args.gpu

    # retrieve folder with test images
    image_path = 'model/data/imagenetfull/test/class'
    assert os.path.exists(image_path), "Input path doesn't exist"
    files = [f for f in listdir(image_path) if isfile(join(image_path, f))]
    print('Number of valid images in folder is:', len(files))

    # allocate space for experimental results
    gzip = []
    bz2 = []
    lzma = []
    png = []
    webp = []
    bbans = []
    bitswap = []

    # set random experiment seed for reproducibility
    np.random.seed(100)

    # randomly draw 1000 images from the test set (without replacement)
    randindices = np.random.choice(len(files), size=1000, replace=False)
    nexperiments = 100

    # some images in the test set have more color channels than 3
    # so we need to track how many valid images we compressed
    compressed = 0

    # compress 100 valid images ("valid" being images with 3 color channels)
    for indx, i in enumerate(randindices):
        # extract 32x32 pixel-blocks from the image
        blocks, h, w, img_valid = convert_image_to_numpy(path=join(image_path, files[i]))
        if img_valid:
            # verbose
            print(f"Compressing image {compressed + 1} with benchmark compressors, BB-ANS and Bit-Swap")

            # reconstruct image from extracted 32x32 pixel-blocks from the image
            cropped_img = np.expand_dims(unextract_blocks(blocks, h, w), 0)

            # apply benchmark compressors
            gzip.append(bench_compressor(gzip_compress, cropped_img))
            bz2.append(bench_compressor(bz2_compress, cropped_img))
            lzma.append(bench_compressor(lzma_compress, cropped_img))
            png.append(bench_compressor(
                pimg_compress("PNG", optimize=True), cropped_img))
            webp.append(bench_compressor(
                pimg_compress('WebP', lossless=True, quality=100), cropped_img))

            # apply BB-ANS and Bit-Swap
            bbans.append(compress(quantbits=10, nz=4, bitswap=0, gpu=gpu, blocks=blocks))
            bitswap.append(compress(quantbits=10, nz=4, bitswap=1, gpu=gpu, blocks=blocks))

            # count number of "valid" compressed images
            compressed += 1
        if compressed == nexperiments:
            break

    # calculate average bitrates over the different experiments
    print(f"gzip: {np.mean(gzip):.2f} bits/dim")
    print(f"bz2: {np.mean(bz2):.2f} bits/dim")
    print(f"lzma: {np.mean(lzma):.2f} bits/dim")
    print(f"png: {np.mean(png):.2f} bits/dim")
    print(f"webp: {np.mean(webp):.2f} bits/dim")
    print(f"bbans: {np.mean(bbans):.2f} bits/dim")
    print(f"bitswap: {np.mean(bitswap):.2f} bits/dim")