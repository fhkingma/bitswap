#Bit-Swap

Code for reproducing results of [Bit-Swap: Practical Lossless Compression with Recursive Bits Back Coding]().

The code is written by [Friso H. Kingma](https://www.linkedin.com/in/friso-kingma-b94496a0/). The paper is written by [Friso H. Kingma](https://www.linkedin.com/in/friso-kingma-b94496a0/), [Pieter Abbeel](https://people.eecs.berkeley.edu/~pabbeel/) and [Jonathan Ho](http://www.jonathanho.me/).

1. [Introduction](##Introduction)
2. [Overview](##Overview)
3. [Requirements](##Requirements)
4. [Launch](##Launch)
    1. [Training](###Model training)
        1. [MNIST](#####MNIST (on 1 GPU))
        2. [CIFAR-10](#####CIFAR-10 (on 8 GPU's with OpenMPI + Horovod))
        3. [ImageNet (32x32)]()
    2. [Compression](###Compression)
        1. [Pre-trained model checkpoints](#####Pre-Trained model checkpoints)
        2. [MNIST](#####MNIST)
        3. [CIFAR-10](#####CIFAR-10)
        4. [ImageNet (32x32)](#####ImageNet (32x32))
    3. [Plots](###Plots)
5. [Citation](##Citation)
6. [Questions](##Questions)
7. [Credits](##Credits and Acknowledgements)

##Introduction
We present a lossless compression scheme, called Bit-Swap, that results in compression rates that are empirically superior to existing techniques. Our work builds on [BB-ANS](https://github.com/bits-back/bits-back) that was originally proposed in [(Townsend et al., 2018)](https://arxiv.org/abs/1901.04866). BB-ANS exploits the combination of the ''bits back'' argument [(Hinton & Van Camp, 1993)](http://www.cs.toronto.edu/~fritz/absps/colt93.pdf), latent variable models and the entropy encoding technique Asymmetric Numeral Systems (ANS) [(Duda, 2009)](https://arxiv.org/abs/0902.0271). We expanded BB-ANS to hierarchical latent variable models, that are known to be better density estimators. When considering the average net bitrate (explained in Section 2.3 in the paper), Bit-Swap outperforms other benchmark lossless compression schemes on [MNIST](http://yann.lecun.com/exdb/mnist/), [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [ImageNet (32x32)](http://image-net.org/small/download.php). The table below shows the performance measured in bits per dimension (bits/dim).

|              | MNIST | CIFAR-10 | ImageNet (32x32) |
|--------------|-------|----------|------------------|
| Uncompressed | 8.00  | 8.00     | 8.00             |
| gzip         | 1.63  | 7.36     | 7.31             |
| bz2          | 1.41  | 6.94     | 6.99             |
| LZMA         | 1.39  | 6.06     | 6.09             |
| PNG          | 2.80  | 5.87     | 6.39             |
| WebP         | 2.10  | 4.61     | 5.29             |
| **Bit-Swap** | **1.27**|**3.79**|**4.48**          |

##Overview
The repository consists of two main parts:
- Training of the variational autoencoders
- Compression with Bit-Swap and BB-ANS using the trained models

Scripts relating to **training of the models** on MNIST ([mnist_train.py](https://github.com/fhkingma/bitswap/blob/master/model/mnist_train.py)), CIFAR-10 ([cifar_train.py](https://github.com/fhkingma/bitswap/blob/master/model/cifar_train.py)) and ImageNet (32x32) ([imagenet_train.py](https://github.com/fhkingma/bitswap/blob/master/model/imagenet_train.py)) can be found in the subdirectory [model](https://github.com/fhkingma/bitswap/tree/master/model). Scripts relating to **compression with Bit-Swap and BB-ANS** of MNIST ([mnist_compress.py](https://github.com/fhkingma/bitswap/blob/master/mnist_compress.py)), CIFAR-10 ([cifar_compress.py](https://github.com/fhkingma/bitswap/blob/master/cifar_compress.py)) and ImageNet (32x32) ([imagenet_compress.py](https://github.com/fhkingma/bitswap/blob/master/imagenet_compress.py)) are in the top directory. The script for compression using the benchmark compressors ([benchmark_compress.py]()) and the script for discretization of the latent space ([discretization.py](https://github.com/fhkingma/bitswap/blob/master/discretization.py)) can also be found in the top directory.

##Requirements
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

Run
```
pip install -r requirements.txt
```

Installation instructions for OpenMPI + Horovod are available on the [github page of Horovod](https://github.com/horovod/horovod).

##Launch

###Model training
#####MNIST (on 1 GPU)
######8 latent layers
```
python mnist_train.py --nz=8 --width=61
```
######4 latent layers
```
python mnist_train.py --nz=4 --width=62
```
######2 latent layers
```
python mnist_train.py --nz=2 --width=63
```
######1 latent layer
```
python mnist_train.py --nz=1 --width=64
```
#####CIFAR-10 (on 8 GPU's with OpenMPI + Horovod)
######8 latent layers
```
mpiexec -np 8 cifar_train.py --nz=8 --width=252
```
######4 latent layers
```
mpiexec -np 8 python cifar_train.py --nz=4 --width=254
```
######2 latent layers
```
mpiexec -np 8 python cifar_train.py --nz=2 --width=255
```
######1 latent layer
```
mpiexec -np 8 python cifar_train.py --nz=1 --width=256
```
#####ImageNet (32x32) (on 8 GPU's with OpenMPI + Horovod)
######4 latent layers
```
mpiexec -np 8 python imagenet_train.py --nz=4 --width=254
```
######2 latent layers
```
mpiexec -np 8 python imagenet_train.py --nz=2 --width=255
```
######1 latent layer
```
mpiexec -np 8 python imagenet_train.py --nz=1 --width=256
```
###Compression
#####Pre-Trained model checkpoints
You can download our pretrained (PyTorch) models used in the paper [here](http://www.fhkingma.com/bitswap/params.zip).

#####MNIST
######8 latent layers
```
python mnist_compress.py --nz=8 --bitswap=1
```
```
python mnist_compress.py --nz=8 --bitswap=0
```
######4 latent layers
```
python mnist_compress.py --nz=4 --bitswap=1
```
```
python mnist_compress.py --nz=4 --bitswap=0
```
######2 latent layers
```
python mnist_compress.py --nz=2 --bitswap=1
```
```
python mnist_compress.py --nz=2 --bitswap=0
```
#####CIFAR-10
######8 latent layers
```
python cifar_compress.py --nz=8 --bitswap=1
```
```
python cifar_compress.py --nz=8 --bitswap=0
```
######4 latent layers
```
python cifar_compress.py --nz=4 --bitswap=1
```
```
python cifar_compress.py --nz=4 --bitswap=0
```
######2 latent layers
```
python cifar_compress.py --nz=2 --bitswap=1
```
```
python cifar_compress.py --nz=2 --bitswap=0
```
#####ImageNet (32x32)
######4 latent layers
```
python imagenet_compress.py --nz=4 --bitswap=1
```
```
python imagenet_compress.py --nz=4 --bitswap=0
```
######2 latent layers
```
python imagenet_compress.py --nz=2 --bitswap=1
```
```
python imagenet_compress.py --nz=2 --bitswap=0
```

###Benchmark compressors
```
python benchmark_compress.py
```

###Plots
#####Cumulative Moving Averages (CMA) of the compression results
```
python cma.py
```

#####Stack plot of the different latent layers
```
python stackplot.py
```

##Citation
```
citation
```

##Questions
Please contact Friso Kingma ([fhkingma@gmail.com](mailto:fhkingma@gmail.com)) if you have any questions.

##Credits and Acknowledgements
