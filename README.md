# Bit-Swap

Code for reproducing results of [Bit-Swap: Recursive Bits-Back Coding for Lossless Compression with Hierarchical Latent Variables](https://arxiv.org/abs/1905.06845), appearing at [International Conference on Machine Learning 2019](https://icml.cc/).

The code is written by [Friso H. Kingma](https://www.fhkingma.com/). The paper is written by [Friso H. Kingma](https://www.fhkingma.com/), [Pieter Abbeel](https://people.eecs.berkeley.edu/~pabbeel/) and [Jonathan Ho](http://www.jonathanho.me/).


## Introduction
We present a lossless compression scheme, called Bit-Swap, that results in compression rates that are empirically superior to existing techniques. Our work builds on [BB-ANS](https://github.com/bits-back/bits-back) that was originally proposed by [Townsend et al, 2018](https://arxiv.org/abs/1901.04866). BB-ANS exploits the combination of the ''bits back'' argument [(Hinton & Van Camp, 1993)](http://www.cs.toronto.edu/~fritz/absps/colt93.pdf), latent variable models and the entropy encoding technique Asymmetric Numeral Systems (ANS) [(Duda, 2009)](https://arxiv.org/abs/0902.0271). We expanded BB-ANS to hierarchical latent variable models, that are known to be better density estimators. Bit-Swap outperforms other benchmark lossless compression schemes on [MNIST](http://yann.lecun.com/exdb/mnist/), [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [ImageNet (32x32)](http://image-net.org/small/download.php). The table below shows the performance measured in bits per dimension (bits/dim).

|              | MNIST | CIFAR-10 | ImageNet (32x32) |
|--------------|-------|----------|------------------|
| Uncompressed | 8.00  | 8.00     | 8.00             |
| [GNU gzip](http://www.gnu.org/home.en.html)         | 1.65  | 7.37     | 7.31             |
| [bzip2](http://www.bzip.org/)          | 1.59  | 6.98     | 7.00             |
| [LZMA](https://www.7-zip.org/)         | 1.49  | 6.09     | 6.15             |
| [PNG](http://www.libpng.org/pub/png/)          | 2.80  | 5.87     | 6.39             |
| [WebP](https://en.wikipedia.org/wiki/WebP)         | 2.10  | 4.61     | 5.29             |
| [BB-ANS](https://github.com/bits-back/bits-back)       | 1.48  | 4.19     | 4.66             |
| **Bit-Swap** | **1.29**|**3.82**|**4.48**          |

## Overview
The repository consists of two main parts:
- Training of the variational autoencoders
- Compression with Bit-Swap and BB-ANS using the trained models

Scripts relating to **training of the models** on 
- MNIST ([``mnist_train.py``](https://github.com/fhkingma/bitswap/blob/master/model/mnist_train.py))
- CIFAR-10 ([``cifar_train.py``](https://github.com/fhkingma/bitswap/blob/master/model/cifar_train.py))
- ImageNet (32x32) ([``imagenet_train.py``](https://github.com/fhkingma/bitswap/blob/master/model/imagenet_train.py)) 

can be found in the subdirectory [``/model``](https://github.com/fhkingma/bitswap/tree/master/model). 

Scripts relating to **compression with Bit-Swap and BB-ANS** of 
- MNIST ([``mnist_compress.py``](https://github.com/fhkingma/bitswap/blob/master/mnist_compress.py))
- CIFAR-10 ([``cifar_compress.py``](https://github.com/fhkingma/bitswap/blob/master/cifar_compress.py))
- ImageNet (32x32) ([``imagenet_compress.py``](https://github.com/fhkingma/bitswap/blob/master/imagenet_compress.py)) 

are in the top directory. The script for compression using the benchmark compressors ([``benchmark_compress.py``]()) and the script for discretization of the latent space ([``discretization.py``](https://github.com/fhkingma/bitswap/blob/master/discretization.py)) can also be found in the top directory.

## Requirements
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

## Launch

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
###### Prepare ImageNet (32x32)
First download the downsized version of ImageNet [here](http://image-net.org/small/download.php). Unpack the train and validation set directories in the directory ``model/data/imagenet/``. After that, run
```
python create_imagenet.py
```
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
### Compression
##### Pre-Trained model checkpoints
We will release the pretrained (PyTorch) models soon.

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

### Benchmark compressors
```
python benchmark_compress.py
```

### Plots
##### Cumulative Moving Averages (CMA) of the compression results
```
python cma.py
```

##### Stack plot of the different latent layers
```
python stackplot.py
```

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

## Questions
Please contact Friso Kingma ([fhkingma@gmail.com](mailto:fhkingma@gmail.com)) if you have any questions.