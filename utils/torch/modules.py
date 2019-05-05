from contextlib import contextmanager

import torch.nn.functional as F
from torch.nn import Module, Parameter, Sequential, Dropout, ELU
from torch.nn import init
from PIL import Image
import os
import torch
import numpy as np
from torch.utils.data import Dataset

_WN_INIT_STDV = 0.05
_SMALL = 1e-10

_INIT_ENABLED = False

@contextmanager
def init_mode():
    global _INIT_ENABLED
    assert not _INIT_ENABLED
    _INIT_ENABLED = True
    yield
    _INIT_ENABLED = False

# PyTorch module that applies Data Dependent Initialization + Weight Normalization
class WnModule(Module):
    """
    Module with data-dependent initialization
    """

    def __init__(self):
        super().__init__()

    def _init(self, *args, **kwargs):
        """
        Data-dependent initialization. Will be called on the first forward()
        """
        raise NotImplementedError

    def _forward(self, *args, **kwargs):
        """
        The standard forward pass
        """
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """
        Calls _init (with no_grad) if not initialized.
        If initialized already, calls _forward.
        """
        if _INIT_ENABLED:
            with torch.no_grad():  # no gradients for the init pass
                return self._init(*args, **kwargs)
        return self._forward(*args, **kwargs)

# Data-Dependent Initialization + Weight Normalization extension of a "Conv2D" module of PyTorch
class WnConv2d(WnModule):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding, init_scale=1.0, loggain=True, bias=True):
        super().__init__()
        self.in_dim, self.out_dim, self.kernel_size, self.stride, self.padding = in_dim, out_dim, kernel_size, stride, padding
        self.bias = bias
        self.init_scale = init_scale
        self.loggain = loggain
        self.v = Parameter(torch.Tensor(out_dim, in_dim, self.kernel_size, self.kernel_size))
        self.gain = Parameter(torch.Tensor(out_dim))
        self.b = Parameter(torch.Tensor(out_dim), requires_grad=True if self.bias else False)

        init.normal_(self.v, 0., _WN_INIT_STDV)
        if self.loggain:
            init.zeros_(self.gain)
        else:
            init.ones_(self.gain)
        init.zeros_(self.b)

    def _init(self, x):
        # calculate unnormalized activations
        y_bchw = self._forward(x)
        assert len(y_bchw.shape) == 4 and y_bchw.shape[:2] == (x.shape[0], self.out_dim)

        # set g and b so that activations are normalized
        y_c = y_bchw.transpose(0, 1).reshape(self.out_dim, -1)
        m = y_c.mean(dim=1)
        s = self.init_scale / (y_c.std(dim=1) + _SMALL)
        assert m.shape == s.shape == self.gain.shape == self.b.shape

        if self.loggain:
            loggain = torch.clamp(torch.log(s), min=-10., max=None)
            self.gain.data.copy_(loggain)
        else:
            self.gain.data.copy_(s)

        if self.bias:
            self.b.data.sub_(m * s)

        # forward pass again, now normalized
        return self._forward(x)

    def _forward(self, x):
        if self.loggain:
            g = softplus(self.gain)
        else:
            g = self.gain
        vnorm = self.v.view(self.out_dim, -1).norm(p=2, dim=1)
        assert vnorm.shape == self.gain.shape == self.b.shape
        w = self.v * (g / (vnorm + _SMALL)).view(self.out_dim, 1, 1, 1)
        return F.conv2d(x, w, self.b, stride=self.stride, padding=self.padding)

    def extra_repr(self):
        return 'in_dim={}, out_dim={}, kernel_size={}, stride={}, padding={}, init_scale={}, loggain={}'.format(self.in_dim, self.out_dim, self.kernel_size, self.stride, self.padding, self.init_scale, self.loggain)

# numerically stable version of the "softplus" function
def softplus(x):
    ret = -F.logsigmoid(-x)
    return ret

# class used to store two sets of parameters
# 1. parameters that are the result of EMA (for evaluation)
# 2. parameters not affected by EMA (for training)
# and to apply EMA to (1.)
class EMA(Module):
    def __init__(self, mu):
        super(EMA, self).__init__()
        # decay parameter
        self.mu = mu

        # parameters affected by EMA
        self.shadow = {}

        # "default" parameters
        self.default = {}

    # set parameters affected by EMA
    def register_ema(self, name, val):
        self.shadow[name] = val.clone()

    # set "default parameters
    def register_default(self, name, val):
        self.default[name] = val.clone()

    # return parameters affected by EMA
    def get_ema(self, name):
        assert name in self.shadow
        return self.shadow[name].clone()

    # return "default" parameters
    def get_default(self, name):
        assert name in self.default
        return self.default[name].clone()

    # apply exponential moving average on parameters stored in self.shadow
    def forward(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average

# PyTorch module that is used to only pass through values
class Pass(Module):
    def __init__(self):
        super(Pass, self).__init__()

    def forward(self, x):
        return x

    def inverse(self, x):
        return x

# PyTorch module used to squeeze from [C, H, W] to [C * factor^2, H // factor, W // factor]
class Squeeze2d(Module):
    def __init__(self, factor=2):
        super(Squeeze2d, self).__init__()
        assert factor >= 2
        self.factor = factor

    def forward(self, x):
        if self.factor == 1:
            return x
        shape = x.shape
        height = int(shape[2])
        width = int(shape[3])
        n_channels = int(shape[1])
        assert height % self.factor == 0 and width % self.factor == 0
        x = x.view(-1, n_channels, height//self.factor, self.factor, width//self.factor, self.factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(-1, n_channels*self.factor*self.factor, height//self.factor, width // self.factor)
        return x

    def extra_repr(self):
        return 'factor={}'.format(self.factor)

# PyTorch module used to squeeze from [C, H, W] to [C / factor^2, H * factor, W * factor]
class UnSqueeze2d(Module):
    def __init__(self, factor=2):
        super(UnSqueeze2d, self).__init__()
        assert factor >= 2
        self.factor = factor

    def forward(self, x):
        if self.factor == 1:
            return x
        shape = x.shape
        height = int(shape[2])
        width = int(shape[3])
        n_channels = int(shape[1])
        x = x.view(-1, int(n_channels/self.factor**2), self.factor, self.factor, height, width)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(-1, int(n_channels/self.factor**2), int(height*self.factor), int(width*self.factor))
        return x

    def extra_repr(self):
        return 'factor={}'.format(self.factor)

# PyTorch module used to build a ResNet layer
class ResNetLayer(Module):
    def __init__(self, inchannels, outchannels, kernel_size=3, stride=1, padding=1, dropout_p=0., act=ELU()):
        super(ResNetLayer, self).__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.dropout_p = dropout_p
        self.stride = stride
        self.act = act

        self.conv1 = WnConv2d(inchannels, outchannels, kernel_size=kernel_size, stride=1,
                                      padding=padding, init_scale=1.0, loggain=True)
        self.dropout = Dropout(dropout_p)
        self.conv2 = WnConv2d(outchannels,  outchannels, kernel_size=kernel_size,
                                      stride=1, padding=padding, init_scale=0., loggain=False)

    def forward(self, x):
        # first convolution preceded and followed by an activation
        c1 = self.act(self.conv1(self.act(x)))

        # dropout layer
        if self.dropout_p > 0.:
            c1 = self.dropout(c1)

        # second convolution
        c2 = self.conv2(c1)

        # residual connection
        return x + c2

# PyTorch module used to build a sequence of ResNet layers
class ResNetBlock(Sequential):
    def __init__(self, inchannels, outchannels, kernel_size=3, stride=1, padding=1, nlayers=1, dropout_p=0.,
                 act=ELU()):
        super(ResNetBlock, self).__init__()
        for i in range(nlayers):
            layer = ResNetLayer(inchannels, outchannels, kernel_size, stride, padding, dropout_p, act)
            self.add_module('res{}layer{}'.format(inchannels, i + 1), layer)

# PyTorch Dataset class custom built for the ImageNet dataset (including applying data pre-processing transforms)
class ImageNet(Dataset):
    def __init__(self, root, file, transform=None):
        self.transform = transform
        self.dir = os.path.join(root, file)
        self.dataset = np.load(self.dir)

    def __getitem__(self, index):
        img = self.dataset[index]

        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.dataset)


def main():
    global _INIT_ENABLED
    print('Outside:', _INIT_ENABLED)
    with init_mode():
        print('Inside:', _INIT_ENABLED)
    print('Outside:', _INIT_ENABLED)
