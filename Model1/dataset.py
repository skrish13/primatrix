import csv
import pickle
import random
import imageio

import numpy as np

from torch.utils.data import Dataset
from skimage.transform import resize

def preprocess(img, shape=None):
	
	if shape:
		img = resize(img, shape)

	if img.max() > 1.:
		img = img / 255.
	img = img.astype(np.float32)

	return img

def make_dataset(classes_gt, vid_dir, shape=(30,64,64,3)):
	
	items = []

	for key,values in classes_gt.items():
		print "in key - ", key
		for value in values:
			vidreader = imageio.get_reader(vid_dir + value, 'ffmpeg')
			vid = np.zeros(shape)
			for num,frame in enumerate(vidreader):
				vid[num] = frame

			vid = vid.transpose(3,0,1,2)		#### Since we need CxDxHxW
			items.append([vid, key])

	savename = "../data/items.pkl"
	pickle.dump(items, open(savename, 'w'), -1)
	
	return items

def load_dataset():
	
	loadname = "../data/items.pkl"
	items = pickle.load(open(loadname))

	return items

def get_test_ids():
	
	test_names = []
	with open('../submission_format.csv','rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		num = 0
		for row in spamreader:
			if num ==0:
				num+=1
				continue

			test_names.append(row[0])

	return test_names

class MatrixDataset(Dataset):
	"""docstring for prematrix"""
	def __init__(self, base_dir, num_classes, mode='train'):
		super(MatrixDataset, self).__init__()
		
		self.num_classes = num_classes
		self.base_dir = base_dir
		self.vid_dir = '../micro/'
		self.shape = (30,64,64,3)
		self.mode = mode

		if mode == 'train':
			self.list_classes_gt = pickle.load(open(base_dir + 'list_classes_gt.pkl'))
			self.gt_dict = pickle.load(open(base_dir + 'gt_dict.pkl'))
			self.ids = self.gt_dict.keys()
		elif mode == 'test':
			self.ids = get_test_ids()
			print len(self.ids)

		# self.itms = make_dataset(self.list_classes_gt, self.vid_dir)

	def __getitem__(self, index):
		
		# vid, target = self.itms[index]

		if self.mode == 'train':
			vidid = self.ids[index]
			target = self.gt_dict[vidid]
			if target == 1:
				if random.random() > 0.5:
					return None, None

			vidreader = imageio.get_reader(self.vid_dir + vidid, 'ffmpeg')
			vid = np.zeros(self.shape, dtype=np.float32)
			for num,frame in enumerate(vidreader):
				frame = preprocess(frame)
				vid[num] = frame

			vid = vid.transpose(3,0,1,2)		#### Since we need CxDxHxW		
			

			return vid, target

		else:
			vidid = self.ids[index]
			vidreader = imageio.get_reader(self.vid_dir + vidid, 'ffmpeg')
			vid = np.zeros(self.shape, dtype=np.float32)
			for num,frame in enumerate(vidreader):
				frame = preprocess(frame)
				vid[num] = frame

			vid = vid.transpose(3,0,1,2)		#### Since we need CxDxHxW

			return vid, [vidid]

	def __len__(self):
		
		# return len(self.itms)
		return len(self.ids)*0.001