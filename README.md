# Unified Hypersphere Embedding for Speaker Recognition
By Mahdi Hajibabaei and Dengxin Dai

### Introduction

This repository contains the code and instruction on how to replicate the experiments done in paper:  [Unified Hypersphere Embedding for Speaker Recognition](https://arxiv.org/abs/1807.08312)

In this work we first train a ResNet-20 with the typical softmax loss function and then depricate the final fully connected layer and fine-tune the rest of network with discriminative loss function such as A-Softmax, AM-Softmax and logistic margin.

### Requirements
1. [*VoxCeleb*](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) dataset and lists of [dataset split](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt) and [verification evaluation pairs](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt)
2. [Sphereface's Caffe build](https://github.com/wy1iu/sphereface/tree/master/tools/caffe-sphereface)
3. [AM-Softmax's Caffe build](https://github.com/happynear/caffe-windows/tree/504d8a85f552e988fabff88b026f2c31cb778329)
4. [Anaconda](https://anaconda.org/anaconda/python) or a similar package that includes NumPy, SciPy and scikit-learn.

### Setup

1. Request the audio files from *Nagrani et al.* and extract the wav files to a directory that would be refered to as *base_address*.

2. Follow the instructions to *make* the caffe and pycaffe for each aforementioned caffe build, since they contain all the modules in a typical Caffe build there is no need to have Caffe pre-installed on your computer.
Add Pycaffe's path to the *PYTHONPATH* environment variable by putting the following to the .bashrc "export PYTHONPATH={PATH_TO_CAFFE}/python"

3. Clone this directory

### Training with augmentation

1. Comment the block of the code in *trainAug.py* that initializes the network's coefficient with *net_weights* and train the network from scratch
with softmax and cross-entropy loss function by setting the argument of caffe.SGDSolver to "prototxt/ResNet-20_solver.prototxt" and executing the *trainAug.py* script

Note: if you are not using Sun Grid Engine, set allocated_GPU to the id of the GPU that you wish to use for training.

2. Uncomment the weight initialization block and set *net_weights* to address of Caffemodel file that resulted in previous step.


