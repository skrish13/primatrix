import csv

from torch.autograd import Variable
from torch.utils.data import DataLoader

import densenet
from densenet import torch, nn, F
from dataset import *

### Dataset configs ###
num_classes = 24

use_cuda = torch.cuda.is_available()

### trained model configs ###
modelname = "3d_model"
trained_path = 'Weights/mc-cls/model_best.pth.tar'
batch_size = 256


if __name__ == '__main__':

	net = densenet.densenet201(sample_size=64, sample_duration=30, num_classes=num_classes)
	trained_weights = torch.load(trained_path)
	net.load_state_dict(trained_weights['state_dict'])

	test_dataset = MatrixDataset(base_dir='../data/', num_classes=num_classes, mode='test')
	test_loader = DataLoader(test_dataset, batch_size=batch_size)

	output_scores = []

	if use_cuda:
		net.cuda()
	net.eval()

	for batch,(vid, vidname) in enumerate(test_loader):

		print "Processing batch {}/{}".format(batch,len(test_loader))

		if use_cuda:
			vid = Variable(vid, volatile=True).cuda()			
		else:
			vid = Variable(vid, volatile=True)
			

		output = net(vid)
		output = F.softmax(output)
		output = output.data.cpu().numpy()
		print output.shape

		for i in xrange(output.shape[0]):
			output_scores.append([vidname[0][i]]) 		#append a row in output_scores -> ['filename.mp4', 0.0, 0.0, .........]
			output_scores[-1].extend(list(output[i]))

	with open('submission_{}.csv'.format(modelname), 'w') as f:
		writer = csv.writer(f)
		writer.writerow(['filename','bird','blank',
		   'cattle','chimpanzee',
		   'elephant','forest buffalo',
		   'gorilla','hippopotamus',
		   'human','hyena', 'large ungulate',
		   'leopard','lion', 'other (non-primate)',
		   'other (primate)','pangolin','porcupine',
		   'reptile','rodent','small antelope',
		   'small cat','wild dog','duiker', 'hog'])

		for output_score in output_scores:
			writer.writerow(output_score)
