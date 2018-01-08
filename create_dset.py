import csv
import pickle
import random
import imageio
import numpy as np
from skimage.transform import resize

classes = ['bird','blank','cattle','chimpanzee','elephant',
		   'forest buffalo','gorilla','hippopotamus','human','hyena',
		   'large ungulate','leopard','lion','other (non-primate)',
		   'other (primate)','pangolin','porcupine','reptile','rodent',
		   'small antelope','small cat','wild dog','duiker', 'hog']

counts = [ 2386, 122270, 372, 5045, 1085,
		   9, 174, 175, 20005, 10,
		   224, 209, 2, 1876, 
		   20349, 63, 569, 7, 2899,
		   273, 79, 21, 21471, 4557 ]

seed = 6554		   

ignore_list = [1, 8]
#### for positive classes
undersample_number = counts
undersample_number[-2] = 6000
undersample_number[1] = 0
undersample_number[8] = 0 ## assuming we use a nice pretrained model for humans :)


def get_train_labels():
	
	gt_dict = {}								## For easier GT extraction
	list_classes_gt = {}						## For sampling
	for i in xrange(len(classes)):
		list_classes_gt[i] = []

	with open('train_labels.csv','rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		num = 0
		for row in spamreader:
			if num ==0:
				num+=1
				continue
			filename = row[0]
			rr = [float(r) for r in row[1:]]
			if rr.count(1) > 1:
				print 'gg'
			classid = rr.index(1)
			
			gt_dict[filename] = classid
			list_classes_gt[classid].append(filename)

	savename = 'data/gt_dict.pkl'
	pickle.dump(gt_dict, open(savename, 'w'), -1)

	savename = 'data/list_classes_gt.pkl'
	pickle.dump(list_classes_gt, open(savename, 'w'), -1)

def get_test_ids():
	
	test_names = []
	with open('submission_format.csv','rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		num = 0
		for row in spamreader:
			if num ==0:
				num+=1
				continue

			test_names.append(row[0])

	return test_names

def preprocess(img, shape=None):
	
	if shape:
		img = resize(img, shape)

	if img.max() > 1.:
		img = img / 255.
	img = img.astype(np.float32)

	return img

def make_dataset(vid_dir='micro/', shape=(30,64,64,3)):
	
	classes_gt = pickle.load(open('data/list_classes_gt.pkl'))
	items = []

	for key,values in classes_gt.items():
		print "In class - ", key
		if key in ignore_list:
			print "Ignoring this class: ", classes[key]
			continue
		if key == 22:
			random.seed(seed)
			values = random.sample(values, undersample_number[key])	

		for num,value in enumerate(values):
			
			if num%500 == 0:
				print "Processing {}/{} videos".format(num,len(values))
			
			vidreader = imageio.get_reader(vid_dir + value, 'ffmpeg')
			vid = np.zeros(shape)
			for num,frame in enumerate(vidreader):
				vid[num] = frame

			vid = vid.transpose(3,0,1,2)		#### Since we need CxDxHxW
			items.append([vid, key])

	savename = "../data/items.pkl"
	pickle.dump(items, open(savename, 'w'), -1)
	
	return items



if __name__ == '__main__':
	
	# get_train_labels()
	make_dataset()
