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
Add Pycaffe's path to the *PYTHONPATH* environment variable by putting the following to the .bashrc: 

export PYTHONPATH={PATH_TO_CAFFE}/python

3. Clone this directory

### Training with augmentation

Note: Since there is an intersection between the training, validation and testing split for verification and identification. You need to specify which task you want to train for by setting *mode* to either 'identification' or 'verification'. If you want to use this pipeline to train on another dataset please modify the content of parse_list to append the address of each sample to *_set and its label to *_label.
Note: if you are not using Sun Grid Engine, set allocated_GPU to the id of the GPU that you wish to use.

1. Comment the block of the code in *trainAug.py* that initializes the network's coefficient with *net_weights* and train the network from scratch
with softmax and cross-entropy loss function by setting the argument of caffe.SGDSolver to "prototxt/ResNet-20_solver.prototxt" and executing the *trainAug.py* script

After training is finished the networks' coefficient and the state of solver would be stored in result/, we will use the network coefficient (result/ResNet-20_512D_iter_61600.caffemodel) to initialize the network for training with more discriminative loss functions.

2. Uncomment the block of the code responsible for fine-tuning that loads the solver of more discriminative loss function and initializes the network with coefficients of softmax with cross entropy trained network. Run the script *trainAug.py* again, this time "LM_512D_30_iter_61600.caffemodel" would be saved to result/.

3. If you have chosen the mode identification, you need to execute *testIdent.py* script. But before executing, please set the variable *net_weights* to caffemodel that you wish to evaluate and *net_prototxt* to prototxt file of structure of network in interest. Run the script, in the end of evaluation the top-1 and top-5 accuracy would be printed in the terminal similar to message below:

The top1 accuracy on test set is 0.9447
The top5 accuracy on test set is 0.9830

4. If you wish to evaluate the verification accuracy of a model trained for the task of verification, you first need to extract the embeddings of utterances within the test set. In order to do so, open the *embedVerif.py* and set the *net_weights* to caffemodel that you wish to evaluate and *net_prototxt* to prototxt file of structure of network of interest. Remember to  set *embedding_file* to a proper name and directory for storing the resulting embedding. Run the *embedVerif.py*, it will take about half an hour to finish on a Titan Xp. The message "Embeddings of test utterances are stored in ..." will presented after successful completion.

In order to evaluate the Equal Error Rate (EER) and minimum of detection cost function on pairs selected by Nagrani et al., set the *embedding_file* in rocVox.py to address of the embedding that you wish to evaluate and execute the script. Two figures will be displayed: The fist on shows the separation of true match (positive) and false match (negative) pairs:
![picture](https://github.com/MahdiHajibabaei/unified-embedding/blob/master/figures/rocVox_pairs.jpeg =250x)


The second figure shows the ROC of the embeddings:


![picture](https://github.com/MahdiHajibabaei/unified-embedding/blob/master/figures/rocVox_ROC.jpeg =250x)



