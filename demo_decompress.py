from model.imagenetcrop_train import Model
from discretization import *
from benchmark_compress import *
import random
import argparse
from tqdm import tqdm
import os
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

def decompress(quantbits, nz, gpu, state, nblocks):
    # model and compression params
    zdim = 8*16*16
    zrange = torch.arange(zdim)
    xdim = 32**2 * 3
    xrange = torch.arange(xdim)
    ansbits = 31 # ANS precision
    type = torch.float64 # datatype throughout compression
    device = "cpu" if gpu < 0 else f"cuda:{gpu}" # gpu

    # set up the different channel dimension
    reswidth = 256

    # <=== MODEL ===>
    model = Model(xs = (3, 32, 32), nz=nz, zchannels=8, nprocessing=4, kernel_size=3, resdepth=8, reswidth=reswidth).to(device)
    model.load_state_dict(
        torch.load(f'model/params/imagenetcrop/nz4',
                   map_location=lambda storage, location: storage
                   )
    )
    model.eval()

    # get discretization bins for latent variables
    zendpoints, zcentres = discretize(nz, quantbits, type, device, model, "imagenet")

    # get discretization bins for discretized logistic
    xbins = ImageBins(type, device, xdim)
    xendpoints = xbins.endpoints()
    xcentres = xbins.centres()

    # < ===== COMPRESSION ===>
    # initialize compression
    model.compress()

    # compression experiment params
    blocks = np.zeros((nblocks, 32, 32, 3), dtype=np.uint8)

    # <===== RECEIVER =====>
    iterator = tqdm(range(nblocks), desc="Decompression")
    for xi in iterator:
        # prior
        cdfs = logistic_cdf(zendpoints[-1].t(), torch.zeros(1, device=device, dtype=type),
                            torch.ones(1, device=device, dtype=type)).t()
        pmfs = cdfs[:, 1:] - cdfs[:, :-1]
        pmfs = torch.cat((cdfs[:, 0].unsqueeze(1), pmfs, 1. - cdfs[:, -1].unsqueeze(1)), dim=1)

        # decode z
        state, zsymtop = ANS(pmfs, bits=ansbits, quantbits=quantbits).decode(state)

        # < ===== Bit-Swap ====>
        # inference and generative model
        for zi in reversed(range(nz)):
            # generative model
            z = zcentres[zi, zrange, zsymtop]
            mu, scale = model.generate(zi)(given=z)
            cdfs = logistic_cdf((zendpoints[zi - 1] if zi > 0 else xendpoints).t(), mu,
                                scale).t()  # most expensive calculation?
            pmfs = cdfs[:, 1:] - cdfs[:, :-1]
            pmfs = torch.cat((cdfs[:, 0].unsqueeze(1), pmfs, 1. - cdfs[:, -1].unsqueeze(1)), dim=1)

            # decode z or x
            state, sym = ANS(pmfs, bits=ansbits, quantbits=quantbits if zi > 0 else 8).decode(state)

            # inference model
            input = zcentres[zi - 1, zrange, sym] if zi > 0 else xcentres[xrange, sym]
            mu, scale = model.infer(zi)(given=input)
            cdfs = logistic_cdf(zendpoints[zi].t(), mu, scale).t()  # most expensive calculation?
            pmfs = cdfs[:, 1:] - cdfs[:, :-1]
            pmfs = torch.cat((cdfs[:, 0].unsqueeze(1), pmfs, 1. - cdfs[:, -1].unsqueeze(1)), dim=1)

            # encode z
            state = ANS(pmfs, bits=ansbits, quantbits=quantbits).encode(state, zsymtop)

            zsymtop = sym

        # reshape to 32x32 pixel-block with 3 color channels
        im = zsymtop.clone().view(3, 32, 32).detach().cpu()
        blocks[blocks.shape[0] - xi - 1] = np.array(im, dtype=np.uint8).transpose((1,2,0))

    return blocks

def input_compressed_file():
    while True:
        sys.stdout.write("Compressed file path: ")
        path = input()

        if not isinstance(path, str):
            print("Path must be string.")
            continue
        if not os.path.exists(path):
            print("Path does not exist.")
            continue
        if not isfile(path):
            print("Path does not point to a file.")
            continue

        dir, file = os.path.split(os.path.abspath(path))
        filename, file_ext = os.path.splitext(file)

        if not file_ext == ".npy":
            print("Extension must be .npy")
            continue

        if not "_bitswap" in filename:
            print("There must be _bitswap at the end of the filename.")
            continue

        state_array = np.load(path)

        if not state_array.dtype == "uint32":
            print("State streams must be 32 bits long.")
            continue

        state = state_array.tolist()

        return state, dir, filename, file_ext

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int, help="-1: use the cpu, [0,1,2,...]: use gpu with specified index")
    args = parser.parse_args()
    gpu = args.gpu

    # seed for replicating experiment and stability
    np.random.seed(100)
    random.seed(50)
    torch.manual_seed(50)
    torch.cuda.manual_seed(50)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # retrieve compressed file
    state, dir, filename, file_ext = input_compressed_file()

    # retrieve back the width, height and number of pixel-blocks
    w = state.pop()
    h = state.pop()
    nblocks = state.pop()

    # move tail bits (everything that was cut off after 32 bits) back to stream
    state.append(state.pop() << 32 | state.pop())

    # decompress with Bit-Swap
    print("")
    blocks = decompress(quantbits=10, nz=4, gpu=gpu, state=state, nblocks=nblocks)

    # reconstruct image and compare
    img_decompressed = unextract_blocks(blocks, h, w)
    img_uncompressed = np.load(join(dir, f"{filename.replace('_bitswap', '_uncompressed')}.npy"))
    assert np.all(img_decompressed == img_uncompressed)

    # save reconstructed image
    im = Image.fromarray(img_decompressed)
    im.save(join(dir, f"{filename.replace('_bitswap', '_recovered')}.jpeg"))

    # verbose
    print("")
    print(f"Reconstructed image as {filename.replace('_bitswap', '_recovered')}.jpeg")
