# Bit-Swap

Code for reproducing results of [Bit-Swap: Recursive Bits-Back Coding for Lossless Compression with Hierarchical Latent Variables](https://arxiv.org/abs/1905.06845), appearing at [International Conference on Machine Learning 2019](https://icml.cc/).

The code is written by [Friso H. Kingma](https://www.fhkingma.com/). The paper is written by [Friso H. Kingma](https://www.fhkingma.com/), [Pieter Abbeel](https://people.eecs.berkeley.edu/~pabbeel/) and [Jonathan Ho](http://www.jonathanho.me/).

## Contents
1. [Introduction](#introduction)
2. [Code overview](#overview)
3. [Requirements](#requirements)
4. [Launch](#launch)
    1. [Model training](#training)
    2. [Compression](#compression)
    2. [Benchmark Compressors](#benchmark)
    3. [Plots](#plots)
5. [Demo (coming soon)](#demo)
6. [Citation](#citation)
7. [Questions](#questions)

<a name="introduction"></a>
## Introduction
We present a lossless compression scheme, called Bit-Swap, that results in compression rates that are empirically superior to existing techniques. Our work builds on [BB-ANS](https://github.com/bits-back/bits-back) that was originally proposed by [Townsend et al, 2019](https://arxiv.org/abs/1901.04866). BB-ANS exploits the combination of the ''bits back'' argument [(Hinton & Van Camp, 1993)](http://www.cs.toronto.edu/~fritz/absps/colt93.pdf), latent variable models and the entropy encoding technique Asymmetric Numeral Systems (ANS) [(Duda et al, 2015)](https://ieeexplore.ieee.org/document/7170048). We extended BB-ANS to be more efficient for hierarchical latent variable models, that are known to be better density estimators.

In one of the experiments, we compressed 100 unscaled and cropped images of ImageNet with Bit-Swap, BB-ANS and other benchmark compressors. Details of the experimental setup can be found [here](#expsetup) or in the [paper](https://arxiv.org/abs/1905.06845). In this regime, Bit-Swap outperforms the other compression schemes. The experimental setup and results of the other experiments can be found in the paper.

| Compression Scheme | Rate (bits/dim) |
|--------------------|-----------------|
| *Uncompressed*       | *8.00*            |
| GNU Gzip           | 5.96            |
| bzip2              | 5.07            |
| LZMA               | 5.09            |
| PNG                | 4.71            |
| WebP               | 3.66            |
| BB-ANS             | 3.62            |
| **Bit-Swap**           | **3.51**            |

<a name="expsetup"></a>
##### Experimental setup: compression of unscaled and cropped ImageNet
For this experiment, we constructed our own train and test set of ImageNet images as described in the instructions [here](#imagenetunscaled). We trained a model on random 32x23 pixel-patches of the constructed train set. Afterwards we 
1. independently took 100 images from the constructed test set
2. cropped the images to multiples of 32 pixels on each side
3. split the images up into grids of 32x32 pixel-blocks
4. compressed the resulting sequence of pixel-blocks of every image with Bit-Swap and BB-ANS and finally
5. calculated the average bitrate over the pixel-blocks of **every image** independently.

We also compressed the cropped images resulting from step 2 with other benchmark compressors.



<a name="overview"></a>
## Overview
**Note**: Before executing any of the scripts, look through the [requirements](#requirements) section and make sure **all** the requirements are satisfied.

The repository consists of two main parts:
<!-- - A demo to compress and decompress your own image. -->
- Training of the variational autoencoders.
- Compression with Bit-Swap and BB-ANS using the trained models.

<!--
Scripts relating to the demo, including
- Compression of your own image to a ``.bitswap`` file ([``image_compress.py``](https://github.com/fhkingma/bitswap/blob/master/image_compress.py))
- Decompression of the resulting ``.bitswap`` file to a reconstruction of your image ([``image_decompress.py``](https://github.com/fhkingma/bitswap/blob/master/image_decompress.py))
-->

Scripts relating to **training of the models** on the training sets of 
- MNIST ([``mnist_train.py``](https://github.com/fhkingma/bitswap/blob/master/model/mnist_train.py))
- CIFAR-10 ([``cifar_train.py``](https://github.com/fhkingma/bitswap/blob/master/model/cifar_train.py))
- ImageNet (32x32) ([``imagenet_train.py``](https://github.com/fhkingma/bitswap/blob/master/model/imagenet_train.py))

and on random 32x32 pixel-patches of
- ImageNet (original size: unscaled and uncropped) ([``imagenetcrop_train.py``](https://github.com/fhkingma/bitswap/blob/master/model/imagenetcrop_train.py)) 

can be found in the subdirectory [``/model``](https://github.com/fhkingma/bitswap/tree/master/model). 

Scripts relating to **compression with Bit-Swap and BB-ANS** of the (partial) test sets of
- MNIST ([``mnist_compress.py``](https://github.com/fhkingma/bitswap/blob/master/mnist_compress.py))
- CIFAR-10 ([``cifar_compress.py``](https://github.com/fhkingma/bitswap/blob/master/cifar_compress.py))
- ImageNet (32x32) ([``imagenet_compress.py``](https://github.com/fhkingma/bitswap/blob/master/imagenet_compress.py)) 

and on 100 images independently taken from the test set of unscaled ImageNet, cropped to multiples of 32 pixels on each side

- ImageNet (unscaled and cropped) ([``imagenetcrop_compress.py``](https://github.com/fhkingma/bitswap/blob/master/imagenetcrop_compress.py))

are in the top directory. The script for compression using the benchmark compressors ([``benchmark_compress.py``](https://github.com/fhkingma/bitswap/blob/master/benchmark_compress.py)) and the script for discretization of the latent space ([``discretization.py``](https://github.com/fhkingma/bitswap/blob/master/discretization.py)) can also be found in the top directory. The script [``imagenetcrop_compress.py``](https://github.com/fhkingma/bitswap/blob/master/imagenetcrop_compress.py) also directly compresses the images with the benchmark compressors.

<a name="requirements"></a>
## Requirements
The following is required to run the scripts:

- Python (3.7)
- OpenMPI and Horovod (0.16.0)
- Numpy (1.15.4)
- PyTorch (1.0.0)
- Torchvision (0.2.1)
- Tensorflow (1.13.1)
- Tensorboard (1.31.1)
- TensorboardX (1.6)
- tqdm (4.28.1)
- Matplotlib (3.0.2)
- Scipy (1.1.0)
- Scikit-learn (0.20.1)

Add the top directory of the repository to the ``$PYTHONPATH`` variable. For example:
```
export PYTHONPATH=$PYTHONPATH:~/bitswap
```

Installation instructions for OpenMPI + Horovod are available on the [github page of Horovod](https://github.com/horovod/horovod).

##### Prepare ImageNet (32x32)
First download the downsized version of ImageNet [here](http://image-net.org/small/download.php). Unpack the train and validation set directories in ``model/data/imagenet/train`` and ``model/data/imagenet/test`` respectively. After that, run
```
python create_imagenet.py
```

<a name="imagenetunscaled"></a>
##### Prepare ImageNet (unscaled)
First download the unscaled ImageNet validation set [here](http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar) and the test set [here](http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_test.tar). Unpack the images of both datasets in ``model/data/imagenet/train/class``. After that, independently take 5000 images from this folder and move them into ``model/data/imagenet/test/class``. In Ubuntu, this can be achieved with the following commands:
```
cd ~/bitswap/model/data/imagenetcrop/train/class
ls | shuf -n 5000 | xargs -i mv {} ~/bitswap/model/data/imagenetcrop/test/class
```

<a name="checkpoints"></a>
##### Pretrained model checkpoints
Pretrained (PyTorch) model checkpoints:
- Download [MNIST model checkpoints](http://fhkingma.com/bitswap/mnist_checkpoints.zip) and unpack in ``/model/params/mnist``
- Download [CIFAR-10 model checkpoints](http://fhkingma.com/bitswap/cifar_checkpoints.zip) and unpack in ``/model/params/cifar``
- Download [ImageNet (32x32) model checkpoints](http://fhkingma.com/bitswap/imagenet_checkpoints.zip) and unpack in ``/model/params/imagenet``
- Download [ImageNet (unscaled) model checkpoints](http://fhkingma.com/bitswap/imaganetcrop_checkpoints.zip) and unpack in ``/model/params/imagenetcrop``

<a name="launch"></a>
## Launch

<a name="training"></a>
### Model training
##### MNIST (on 1 GPU)
###### 8 latent layers
```
python mnist_train.py --nz=8 --width=61
```
###### 4 latent layers
```
python mnist_train.py --nz=4 --width=62
```
###### 2 latent layers
```
python mnist_train.py --nz=2 --width=63
```
###### 1 latent layer
```
python mnist_train.py --nz=1 --width=64
```
##### CIFAR-10 (on 8 GPU's with OpenMPI + Horovod)
###### 8 latent layers
```
mpiexec -np 8 python cifar_train.py --nz=8 --width=252
```
###### 4 latent layers
```
mpiexec -np 8 python cifar_train.py --nz=4 --width=254
```
###### 2 latent layers
```
mpiexec -np 8 python cifar_train.py --nz=2 --width=255
```
###### 1 latent layer
```
mpiexec -np 8 python cifar_train.py --nz=1 --width=256
```
##### ImageNet (32x32) (on 8 GPU's with OpenMPI + Horovod)
###### 4 latent layers
```
mpiexec -np 8 python imagenet_train.py --nz=4 --width=254
```
###### 2 latent layers
```
mpiexec -np 8 python imagenet_train.py --nz=2 --width=255
```
###### 1 latent layer
```
mpiexec -np 8 python imagenet_train.py --nz=1 --width=256
```
##### ImageNet (unscaled) (on 8 GPU's with OpenMPI + Horovod)
###### 4 latent layers
```
mpiexec -np 8 python imagenetcrop_train.py --nz=4 --width=256
```
<a name="compression"></a>
### Compression
##### MNIST
###### 8 latent layers
```
python mnist_compress.py --nz=8 --bitswap=1
```
```
python mnist_compress.py --nz=8 --bitswap=0
```
###### 4 latent layers
```
python mnist_compress.py --nz=4 --bitswap=1
```
```
python mnist_compress.py --nz=4 --bitswap=0
```
###### 2 latent layers
```
python mnist_compress.py --nz=2 --bitswap=1
```
```
python mnist_compress.py --nz=2 --bitswap=0
```
##### CIFAR-10
###### 8 latent layers
```
python cifar_compress.py --nz=8 --bitswap=1
```
```
python cifar_compress.py --nz=8 --bitswap=0
```
###### 4 latent layers
```
python cifar_compress.py --nz=4 --bitswap=1
```
```
python cifar_compress.py --nz=4 --bitswap=0
```
###### 2 latent layers
```
python cifar_compress.py --nz=2 --bitswap=1
```
```
python cifar_compress.py --nz=2 --bitswap=0
```
##### ImageNet (32x32)
###### 4 latent layers
```
python imagenet_compress.py --nz=4 --bitswap=1
```
```
python imagenet_compress.py --nz=4 --bitswap=0
```
###### 2 latent layers
```
python imagenet_compress.py --nz=2 --bitswap=1
```
```
python imagenet_compress.py --nz=2 --bitswap=0
```

##### ImageNet (unscaled & cropped)
```
python imagenetcrop_compress.py
```
<a name="benchmark"></a>
### Benchmark compressors
```
python benchmark_compress.py
```

### Plots <a name="plots"></a>
##### Cumulative Moving Averages (CMA) of the compression results
```
python cma.py
```

##### Stack plot of the different latent layers
```
python stackplot.py
```

## DEMO: Compress your own image with Bit-Swap <a name="demo"></a>
Coming soon.

<a name="citation"></a>
## Citation
If you find our work useful, please cite us in your work.

```
@inproceedings{kingma2019bitswap,
  title={Bit-Swap: Recursive Bits-Back Coding for Lossless Compression with Hierarchical Latent Variables},
  author={Kingma, Friso H and Abbeel, Pieter and Ho, Jonathan},
  booktitle={International Conference on Machine Learning},
  year={2019}
}
```

<a name="questions"></a>
## Questions
Please contact Friso Kingma ([fhkingma@gmail.com](mailto:fhkingma@gmail.com)) if you have any questions.
