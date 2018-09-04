import numpy as np
import caffe
print "Caffe successfully imported!"
import scipy.io.wavfile
import re
import os.path

wav_list='/scratch_net/biwidl09/hmahdi/VoxCeleb/Identification_split.txt'
base_address='/scratch_net/biwidl09/hmahdi/VoxCeleb/voxceleb1_wav/'
pre_emphasis = 0.97
frame_size = 0.025
frame_stride = 0.01
NFFT = 512
BATCH_SIZE=50
train_set=[]
train_label=[]
validation_set=[]
validation_label=[]
test_set=[]
test_label=[]
spectrogram_batch=np.empty([BATCH_SIZE,1,300,257],dtype=float) #np.int((NFFT+1)/2)
label_batch=np.empty([BATCH_SIZE,1,1,1],dtype=float)
MAX_ITERATIONS=61600
TEST_INTERVAL=1000

# Spectrogram extractor, courtesy of Haytham Fayek
def spectrogram_extractor(file_name,i):

	sample_rate, signal = scipy.io.wavfile.read(file_name)  
	extended_signal=np.append(signal,signal)
	beginning=int((len(signal))*np.random.random_sample())
	signal = extended_signal[beginning:beginning+48241]
	if (np.int(np.random.random_sample()*2)==1):
		signal= signal[::-1]
	signal=signal-np.mean(signal)
	emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

	frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate 
	signal_length = len(emphasized_signal)
	frame_length = int(round(frame_length))
	frame_step = int(round(frame_step))
	num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) 

	pad_signal_length = num_frames * frame_step + frame_length
	z = np.zeros((pad_signal_length - signal_length))
	pad_signal = np.append(emphasized_signal, z) 

	indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
	frames = pad_signal[indices.astype(np.int32, copy=False)]

	frames *= np.hamming(frame_length)
	mag_frames = np.absolute(np.fft.rfft(frames, NFFT)) 

	spectrogram_batch[i,0,:,:]= (mag_frames - mag_frames.mean(axis=0)) / mag_frames.std(axis=0)

# This function parses the file list given by Nagrani et al.
def parse_list(mode):
	
	input_file = open(wav_list,'r')
	identity=0
	split_index=1
	if (mode=='verification'):
		for line in input_file:
			parsed_line=re.split(r'[ \n]+', line)	
			utterance_address=base_address+parsed_line[1]
			prev_split_index=split_index
			split_index=int(parsed_line[0])
			if (prev_split_index>split_index):
				identity=identity+1
			if (os.path.isfile(utterance_address)==False): 
				continue
			elif (parsed_line[1][0]=='E'):
				test_set.append(utterance_address)
				test_label.append(identity)
			elif (split_index==2):
				validation_set.append(utterance_address)
				validation_label.append(identity)
			else :#(split_index==1)
				train_set.append(utterance_address)
				train_label.append(identity)

	elif (mode=='identification'):
		for line in input_file:
			parsed_line=re.split(r'[ \n]+', line)	
			utterance_address=base_address+parsed_line[1]
			prev_split_index=split_index
			split_index=int(parsed_line[0])
			if (prev_split_index>split_index):
				identity=identity+1
			if (os.path.isfile(utterance_address)==False): 
				continue
			elif (split_index==1):
				train_set.append(utterance_address)
				train_label.append(identity)
			elif (split_index==2):
				validation_set.append(utterance_address)
				validation_label.append(identity)
			elif (split_index==3):
				test_set.append(utterance_address)
				test_label.append(identity)
	else:
		print "Parsing failed due to undefined mode..."
		return
	print("The size of training set: %d" % (len(train_set))) 
	print("The size of validation set: %d" % (len(validation_set))) 
	print("The size of test set: %d" % (len(test_set))) 
	print("Number of identities: %d" % (identity+1))

def validate_accuracy(solver):
	print "Start of validation process.."
	top1Accuracy=0
	top5Accuracy=0
	validation_set_size=len(validation_set)
	for i in range(0,validation_set_size/BATCH_SIZE):
		for j in range(0,BATCH_SIZE):
			sample_index=i*BATCH_SIZE+j
			file_name=validation_set[sample_index]
			label_batch[j,0,0,0]=validation_label[sample_index]
			spectrogram_extractor(file_name,j)
		solver.net.blobs['data'].data[...] =spectrogram_batch
		solver.net.blobs['label'].data[...]  =label_batch
		solver.net.forward()
		top1Accuracy=top1Accuracy+solver.net.blobs['top1Accuracy'].data
		top5Accuracy=top5Accuracy+solver.net.blobs['top5Accuracy'].data
	top1Accuracy=top1Accuracy*BATCH_SIZE/validation_set_size
	top5Accuracy=top5Accuracy*BATCH_SIZE/validation_set_size	
	print("The top1 accuracy on validation set is %.4f" % (top1Accuracy))
	print("The top5 accuracy on validation set is %.4f" % (top5Accuracy))
		
if __name__ == '__main__':

	mode='identification'
	allocated_GPU= int(os.environ['SGE_GPU'])
	print("The %s training will be executed on GPU #%d" % (mode,allocated_GPU))
	caffe.set_device(allocated_GPU)
	caffe.set_mode_gpu()
	solver = caffe.SGDSolver("prototxt/ResNet-20_solver.prototxt")# Network with typical softmax with cross entropy loss function
	# Uncomment the following block when training from scratch and uncomment to fine-tune
	''''
	solver = caffe.SGDSolver("prototxt/LogisticMargin_solver.prototxt")# Network with logistic margin loss function
	net_weights='result/ResNet-20_512D_iter_61600.caffemodel'
	print("The network will be initialized with %s" % (net_weights))
	solver.net.copy_from(net_weights)
	'''
	parse_list(mode)
	train_set_size=len(train_set)
	
	for j in range(0,MAX_ITERATIONS):
		for i in range(0,BATCH_SIZE):
			sample_index=np.int(np.random.random_sample()*train_set_size)
			file_name=train_set[sample_index] 
			spectrogram_extractor(file_name,i)
			label_batch[i,0,0,0]=train_label[sample_index]
		solver.net.blobs['data'].data[...] =spectrogram_batch
		solver.net.blobs['label'].data[...] =label_batch
		solver.step(1)
		if (np.mod(j,TEST_INTERVAL)==0):
			validate_accuracy(solver)

