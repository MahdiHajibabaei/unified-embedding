
import numpy as np
import re
import os.path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score


if __name__ == '__main__':

	test_set=[]
	test_label=[]
	wav_list='/scratch_net/biwidl09/hmahdi/VoxCeleb/Identification_split.txt'
	base_address='/scratch_net/biwidl09/hmahdi/VoxCeleb/voxceleb1_wav/'
	input_file = open(wav_list,'r')
	identity=0
	split_index=1
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

	print("The size of test set: %d" % (len(test_set))) 
	print("Number of identities: %d" % (identity+1))
	roc_size=4872
	embeddings=np.load("ResNet_512thin.npy")
	embeddings=np.asarray(embeddings[0:roc_size,:])
	embeddings=(embeddings) / np.sqrt(np.sum(np.square(embeddings), axis=1))[:,None]
	score_matrix=[]
	label_matrix=[]
	true_match=[]
	false_match=[]
	
	for i in range(0,roc_size):
		for j in range(0,i):
			score=np.dot(embeddings[i,:],embeddings[j,:])
			if (test_label[i]==test_label[j]):
				if (i!=j):
					score_matrix.append(score)
					label_matrix.append(1)
					true_match.append(score)
			else:
				score_matrix.append(score)
				label_matrix.append(0)
				false_match.append(score)
	print("There are %d matching pairs and %d non-matching pairs" % (len(true_match),len(false_match)))

	bins = np.linspace(-1.2, 1.2, 240)
	plt.hist(true_match, bins,  density=True, facecolor='g', alpha=0.75,label='true_match')
	plt.hist(false_match, bins,  density=True, label='false_match')
	plt.legend(loc='upper right')
	plt.show()

	fpr, tpr, _ = roc_curve(label_matrix,score_matrix)
	intersection=abs(1-tpr-fpr)
	DCF2=100*(0.01*(1-tpr)+0.99*fpr)
	DCF3=1000*(0.001*(1-tpr)+0.999*fpr)
	print("EER= %.2f  DCF0.01= %.3f  DCF0.001= %.3f" %(100*fpr[np.argmin(intersection)],np.min(DCF2),np.min(DCF3)))
	plt.figure()
	plt.semilogx(fpr, tpr)
	plt.grid(True)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic')
	plt.legend(loc="lower right")
	plt.show()


