# Unified Hypersphere Embedding for Speaker Recognition
By Mahdi Hajibabaei and Dengxin Dai

### Introduction

This repository contains the code and instruction to replicate the experiments done in paper: [Unified Hypersphere Embedding for Speaker Recognition](https://arxiv.org/abs/1807.08312)

In this work, first, we train a ResNet-20 with the typical softmax with cross entropy loss function and then fine-tune the network with more discriminative loss function such as A-Softmax, AM-Softmax and logistic margin.

### Requirements
1. [*VoxCeleb*](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) dataset and lists of [dataset split](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt) and [verification evaluation pairs](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt)
2. [Sphereface's Caffe build](https://github.com/wy1iu/sphereface/tree/master/tools/caffe-sphereface)
3. [AM-Softmax's Caffe build](https://github.com/happynear/caffe-windows/tree/504d8a85f552e988fabff88b026f2c31cb778329)
4. [Anaconda](https://anaconda.org/anaconda/python) or a similar package that includes NumPy, SciPy and scikit-learn.

### Setup

1. Request the audio files from *Nagrani et al.* and extract the wav files to a directory that would be refered to as *base_address*.

2. Follow the instructions to *make* the Caffe and Pycaffe for each aforementioned build.
Add Pycaffe's path as *PYTHONPATH* environment variable by copying the following line to the .bashrc: 

	export PYTHONPATH={PATH_TO_CAFFE}/python

3. Clone this repository

### Training with augmentation

   Note: Since there is an intersection between the training, validation and testing split for verification and identification. You need to specify which task you want to train for by setting *mode* to either 'identification' or 'verification'. If you want to use this pipeline to train on another dataset please modify the content of parse_list function to append the address of each sample to *_set and its label to *_label.
   Note: if you are not using Sun Grid Engine, set allocated_GPU to the id of the GPU that you wish to use.

1. Comment the following block of the code in *train_aug.py* that initializes the network's coefficient with *net_weights* and trains the network from scratch
with softmax and cross entropy loss function by setting the argument of caffe.SGDSolver to "prototxt/ResNet-20_solver.prototxt" and executing the *train_aug.py* script.

	solver = caffe.SGDSolver("prototxt/LogisticMargin_solver.prototxt")
	net_weights='result/ResNet-20_512D_iter_61600.caffemodel'
	print("The network will be initialized with %s" % (net_weights))
	solver.net.copy_from(net_weights)

   After training is finished, the trained network's coefficients and the state of solver would be stored in *result* directory, we will use the network coefficients (e.g. result/ResNet-20_512D_iter_61600.caffemodel) to initialize the network for training with more discriminative loss functions.

2. Uncomment the aforementioned block of the code and make sure *net_weights* is set to the address of the previous caffemodel. Run *train_aug.py* again, this time "LM_512D_30_iter_61600.caffemodel" would be saved to *result* directory.

3. If you have chosen the mode identification, you need to execute *test_ident.py* script. But before executing, please set the variable *net_weights* to caffemodel that you wish to evaluate and *net_prototxt* to prototxt file of structure of network in interest. Run the script, in the end of execution, the top-1 and top-5 accuracies would be printed in the terminal similar to the message below:

	The top1 accuracy on test set is 0.9447

	The top5 accuracy on test set is 0.9830

    Important note: in recent face recognition literature, similarity of two face images are evaluated by first projecting each face image into an embedding space by feeding it to a CNN. Then a score is given to the similarity of images based on cosine similarity of the embeddings of two faces. In result, first we need to embed each sample into a relatively low dimensional embedding space ( by executing embed_verif.py) and then we can use cosine similarity of these embedding to evaluate the odds of two utterances belonging to the same person

4. If you wish to evaluate the verification accuracy of a model trained for the task of verification, you first need to extract the embeddings of utterances within the test set. In order to do so, open the *embed_verif.py* and set the *net_weights* to caffemodel that you wish to evaluate and *net_prototxt* to prototxt of the structure of network of interest. Remember to  set *embedding_file* to a proper name and directory for storing the resulting embedding. After executing embed_verif.py, the message "Embeddings of test utterances are stored in ..." will printed in terminal.

    In order to evaluate the Equal Error Rate (EER) and minimum of detection cost function (DCF) on pairs selected by Nagrani et al., set the *embedding_file* in roc_vox.py to address of the embedding that you wish to evaluate and execute the script. Two figures will be displayed: The first one shows the separation between positive pairs and negative pairs:
![picture](https://github.com/MahdiHajibabaei/unified-embedding/blob/master/figures/rocVox_pairs.jpeg)


The second figure shows the ROC of the embeddings:


![picture](https://github.com/MahdiHajibabaei/unified-embedding/blob/master/figures/rocVox_ROC.jpeg)

The EER and minimum of detection cost functions (DCF) would be printed on the console afterwards.

If you wish to evaluate the verification accuracy of any trained model on all possible (11.9 million) pairs within verification test set, set the *embedding_file* in roc.py to address of evaluated embeddings and run the script. Similar to the evaluating on few pre-selected pair, two figures would be shown as follows:

![picture](https://github.com/MahdiHajibabaei/unified-embedding/blob/master/figures/roc_pairs.jpeg)

![picture](https://github.com/MahdiHajibabaei/unified-embedding/blob/master/figures/roc_ROC.jpeg)

### Training and/or evaluating without augmentation

If you wish to compare the prediction accuracy and performance of models trained and/or evaluated without repetition and time-reversion augmentation, alter with following lines:	
	
	extended_signal=np.append(signal,signal)
	beginning=int((len(signal))*np.random.random_sample())
	signal = extended_signal[beginning:beginning+48241]
	if (np.int(np.random.random_sample()*2)==1):
		signal= signal[::-1]

with:

	beginning=int((len(signal)-48241)*np.random.random_sample())
	signal = signal[beginning:beginning+48241]

in train_aug.py if you with to eliminate augmentation in training phase and in embed_verif.py or test_ident.py if you wish to eliminate augmentation in evaluating verification and identification accuracies respectively. keep in mind that 48241 samples represents 3.015 seconds of recordings plus an extra sample to compensate for receptive field of pre-emphasis.

### Effect of dropout on verification accuracy 

Applying dropout to penultimate layer of CNN improved the verification accuracy but deteriorated the identification accuracy. If you wish to apply drop out during training, just uncomment the following lines from prototxt of the network's structure. 

	layer {
	name: "drop6"
	type: "Dropout"
	bottom: "res4_3p"
	top: "res4_3p"
	dropout_param {
		dropout_ratio: 0.5
	  }
	}

### Future works

At the time of running the experiments in this work, *VoxCeleb* was the largest publicly available dataset. But not long after, much larger [*VoxCeleb2*](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) dataset with more speakers and more statistically sound evaluations was released and it would be really interesting to see how much improvement using suggested loss functions and augmentation would yield.

There is also an ongoing National Institute of Standard and Technology Speaker Recognition Evaluation (NIST SRE) challenge that lists *VoxCeleb* and *VoxCeleb2* as their official training dataset. It would be interesting to see how much improvement using the suggested loss functions and augmentation would yield.

### Citation

If you plan to use the repetition and time-reversion augmentation, please consider citing my paper:

	@article{hajibabaei2018unified,
	title={Unified Hypersphere Embedding for Speaker Recognition},
	author={Hajibabaei, Mahdi and Dai, Dengxin},
	journal={arXiv preprint arXiv:1807.08312},
	year={2018}
	}

And if you plan to use logistic margin loss function please cite the original AM-Softmax paper (with bibtex given below) along my paper.

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

