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

**Important note: in recent face recognition literature, similarity of two face images are evaluated by first projecting each face image into an embedding space by feeding it to a CNN. Then a score is given to the similarity of images based on cosine similarity of the embedding of two faces. In result, first we need to embed each sample into a relatively low dimensional embedding space ( by executing embedVerif) and then we can use cosine similarity of these embedding two evaluate the odds of two utterances belonging to the same person**

4. If you wish to evaluate the verification accuracy of a model trained for the task of verification, you first need to extract the embeddings of utterances within the test set. In order to do so, open the *embedVerif.py* and set the *net_weights* to caffemodel that you wish to evaluate and *net_prototxt* to prototxt file of structure of network of interest. Remember to  set *embedding_file* to a proper name and directory for storing the resulting embedding. Run the *embedVerif.py*, it will take about half an hour to finish on a Titan Xp. The message "Embeddings of test utterances are stored in ..." will presented after successful completion.

In order to evaluate the Equal Error Rate (EER) and minimum of detection cost function on pairs selected by Nagrani et al., set the *embedding_file* in rocVox.py to address of the embedding that you wish to evaluate and execute the script. Two figures will be displayed: The first one shows the separation between true match (positive) and false match (negative) pairs:
![picture](https://github.com/MahdiHajibabaei/unified-embedding/blob/master/figures/rocVox_pairs.jpeg)


The second figure shows the ROC of the embeddings:


![picture](https://github.com/MahdiHajibabaei/unified-embedding/blob/master/figures/rocVox_ROC.jpeg)

The EER and minimum of detection cost functions (DCF) would be printed on the console afterwards.

If you wish to evaluate the verification accuracy of any trained model on all possible pairs (about 11.9 million) within verification test set, set the *embedding_file* in roc.py to address of evaluated embeddings and run the script. Similar to the evaluating on few pre-selected pair, two figures would be shown as follows:

![picture](https://github.com/MahdiHajibabaei/unified-embedding/blob/master/figures/roc_pairs.jpeg)

![picture](https://github.com/MahdiHajibabaei/unified-embedding/blob/master/figures/roc_ROC.jpeg)


Since the most computational expensive part of evaluation in verification is embedding the test utterances into embedding space and evaluating ROC's only requires number_of_pairs * embedding_dimensionality multiplication and addition operation, evaluating verification accuracy using all possible pairs won't add a significant computational overhead overall. However, in numerous face recognition literature (such as BLUFR) it has been shown that using very few pairs for evaluating and comparing different implementation will result in unfair and unrealistic optimization. In result, **please evaluate the verification accuracy of your model (whether  recognizing through face or voice) on all possible pairs not few pre-selected ones**.   

### Training and/or evaluating without augmentation

If you wish to compare the prediction accuracy and performance of models trained and/or evlauted without repetition and time-reversion augmentation, alter with following lines:	
	
>>	extended_signal=np.append(signal,signal)
>>	beginning=int((len(signal))*np.random.random_sample())
>>	signal = extended_signal[beginning:beginning+48241]
>>	if (np.int(np.random.random_sample()*2)==1):
>>		signal= signal[::-1]

with:

>>	beginning=int((len(signal)-48241)*np.random.random_sample())
>>	signal = signal[beginning:beginning+48241]

in trainAug.py if you with to eliminate augmentation in training phase and in embedVerif.py or testIdent.py if you wish to eliminate augmentation in evaluating verification and identification accuracies respectively.

The 48241 samples represents 3.015 seconds of recordings plus an extra sample to compensate for receptive field of pre-emphasis.

### Future works

At the time of running this experiments, *VoxCeleb* was the largest publicly available dataset in hand with two weeks of recordings and 40 speakers set aside for testing verification accuracy. however, [*VoxCeleb2*](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) dataset with more speakers and more statistically sound evaluation was released in April 2018. It would be really interesting to see how much improvement using suggested loss functions and augmentation would yield.

There is also an ongoing National Institute of Technology Speaker Recognition Evaluation (NIST SRE) challenge that lists *VoxCeleb* and *VoxCeleb2* as their official training dataset. It would be interesting to see how much of an edge using the suggested loss functions and augmentation would give to the contestant.

### Citation

If you plan to use the repetition and time-reversion augmentation, please consider citing my paper:

@article{hajibabaei2018unified,
  title={Unified Hypersphere Embedding for Speaker Recognition},
  author={Hajibabaei, Mahdi and Dai, Dengxin},
  journal={arXiv preprint arXiv:1807.08312},
  year={2018}
}

And if you plan to use Logistic Margin loss function please cite the original AM-Softmax paper (with bibtex given below) along my paper.

@article{wang2018additive,
  title={Additive margin softmax for face verification},
  author={Wang, Feng and Cheng, Jian and Liu, Weiyang and Liu, Haijun},
  journal={IEEE Signal Processing Letters},
  volume={25},
  number={7},
  pages={926--930},
  year={2018},
  publisher={IEEE}
}

