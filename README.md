# Bit-Swap

Code for reproducing results of [Bit-Swap: Practical Lossless Compression with Recursive Bits Back Coding](), appearing at ICML 2019.

The code is written by [Friso H. Kingma](https://www.linkedin.com/in/friso-kingma-b94496a0/). The paper is written by [Friso H. Kingma](https://www.linkedin.com/in/friso-kingma-b94496a0/), [Pieter Abbeel](https://people.eecs.berkeley.edu/~pabbeel/) and [Jonathan Ho](http://www.jonathanho.me/).

## Introduction
The ''bits back'' argument suggests that latent variable models can be turned into lossless compression schemes. Translating the ''bits back'' argument into efficient and practical lossless compression schemes for general latent variable models, however, is still an open problem. Bits-Back with Asymmetric Numeral Systems ([BB-ANS](https://github.com/bits-back/bits-back)), makes bits back coding practically feasible for latent variable models with one latent layer, but it is inefficient for hierarchical latent variable models. In the paper we propose Bit-Swap, a new compression scheme that generalizes BB-ANS and achieves strictly better compression rates for hierarchical latent variable models with Markov chain structure. Through experiments we verify that our proposed technique results in lossless compression rates that are empirically superior to existing techniques.

## Overview
The repository consists of two main parts:
- Training of the variational-autoencoders
- Compression with Bit-Swap and BB-ANS using the trained models

Scripts relating to **training of the models** on MNIST ([mnist_train.py]()), CIFAR-10 ([cifar_train.py]()) and ImageNet (32x32) ([imagenet_train.py]()) can be found in the subdirectory [model](). Scripts relating to **compression with Bit-Swap and BB-ANS** of MNIST ([mnist_compress.py]()), CIFAR-10 ([cifar_compress.py]()) and ImageNet (32x32) ([imagenet_compress.py]()) are in the top directory. The script for compression using the benchmark compressors ([benchmark_compress.py]()) and the script for discretization of the latent space ([discretization.py]()) can also be found in the top directory.

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

Run
```
pip install -r requirements.txt
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
mpiexec -np 8 cifar_train.py --nz=8 --width=252
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


## Contact
Please contact Friso Kingma ([fhkingma@gmail.com](mailto:fhkingma@gmail.com)) if you have any questions.

## Credits and Acknowledgements
