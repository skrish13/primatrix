import time,os,shutil,sys,visdom
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import densenet
from dataset import *
from densenet import torch, nn

'''
Things to change before running
1. !!!!!!!!!!MODELNAME!!!!!!!!!!!
2. params if necessary
3. resume flag and path if resuming
'''

# Dataset configs #
num_classes = 24
labels = [ i for i in xrange(num_classes)]
sample_size = 335
random_seed = 6554

# Training configs #
epochs = 5
num_workers = 0
use_cuda = torch.cuda.is_available()

# Parameters #
batch_size = 78
lr = 1e-3
momentum = 0.6
weight_decay = 5e-4
early_stopping = 30
min_acc_thresh = 0.

# I/O configs #
basename = 'PREMATRIX_DenseNet_'
modelname = sys.argv[1]
weights_folder = './Weights/{}/'.format(modelname)
plot_folder = './Plots/{}/'.format(modelname)
pretrained = False
if pretrained:
	pretrained_path = '../densenet-201-kinetics.pth'
resume = False
if resume:
	checkpoint_path = './Weights/mc-cls/checkpoint.pth.tar'

vis = visdom.Visdom()

def plot_visdom(epoch,y,win):
	if resume:
		x = np.asarray(range(epoch*len(y), (epoch+1)*len(y)))
		y = np.asarray(y)
		vis.line(Y=y, X=x, win=win, env=basename+modelname, opts=dict(title=win), update='append')
		#TODO plt for resuming
	else:
		x = np.arange(len(y))
		y = np.asarray(y)
		vis.line(Y=y, X=x, win=win, env=basename+modelname, opts=dict(title=win))
		plt.plot(x,y)
		plt.savefig(plot_folder + win + '.jpg')
		plt.clf()

def make_dir(folder):
	if not os.path.exists(folder):
		os.makedirs(folder)

def save_checkpoint(state, is_best, filename=weights_folder+'checkpoint.pth.tar'):
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, weights_folder+'model_best.pth.tar')

def get_class_weights(train_dataset, train_loader):

	num_train = len(train_dataset)
	num_train_class = torch.zeros(num_classes)
	cnt = 0
	for vid, mask, target in train_dataset.itms:
		cnt +=1 
		num_train_class[target] += 1
	
	class_weights = (float(num_train) / num_train_class).cuda()
	class_weights /= class_weights.min()
	print num_train_class
	print class_weights

	return class_weights

def get_train_valid_loader(train_dataset, test_size, shuffle=True):

	num_train = len(train_dataset)
	num_train_class = torch.zeros(num_classes)
	indices = list(range(num_train))
	split = int(np.floor(test_size * num_train))

	if shuffle == True:
		np.random.seed(random_seed)
		np.random.shuffle(indices)

	train_idx, test_idx = indices[split:], indices[:split]

	train_sampler = SubsetRandomSampler(train_idx)
	test_sampler = SubsetRandomSampler(test_idx)

	train_loader = DataLoader(train_dataset, 
					batch_size=batch_size, sampler=train_sampler, 
					num_workers=num_workers)

	# for vid, target in train_loader:
	# 	num_train_class[target.cpu().numpy()[0]] += 1
	# class_weights = 1 - (num_train_class / (num_train - split) ).cuda()
	class_weights = None

	test_loader = DataLoader(train_dataset, 
					batch_size=batch_size, sampler=test_sampler, 
					num_workers=num_workers)
	
	return train_loader, test_loader, class_weights

def train(net, criterion, optimizer, epoch, train_loader, temp_loss_epoch, train_loss, output_labels, target_labels):

	for batch,(vid, target) in enumerate(train_loader):
		print vid.shape
		if use_cuda:
			vid = Variable(vid).cuda()
			target = Variable(target).cuda()
		else:
			vid = Variable(vid)
			target = Variable(target)

		output = net(vid)
		optimizer.zero_grad()
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()

		train_loss.append(loss.cpu().data.numpy()[0])
		temp_loss_epoch.append(loss.cpu().data.numpy()[0])
		print "Epoch: {}, Batch: {}/{}, Loss:{}".format(epoch,batch,len(train_loader),train_loss[-1])

		output_argmax = output.cpu().max(1)[1].data.numpy() #max of dim=1, get arg from the tuple [1]

		output_labels.extend(output_argmax)
		target_labels.extend(target.cpu().data.numpy())

	return train_loss, temp_loss_epoch, output_labels, target_labels

def test(net, criterion, optimizer, epoch, test_loader, temp_loss_epoch, test_loss, output_labels, target_labels):

	for batch,(vid, target) in enumerate(test_loader):

		if use_cuda:
			vid = Variable(vid, volatile=True).cuda()
			target = Variable(target, volatile=True).cuda()
		else:
			vid = Variable(vid, volatile=True)
			target = Variable(target, volatile=True)

		output = net(vid)
		loss = criterion(output, target)

		test_loss.append(loss.cpu().data.numpy()[0])
		temp_loss_epoch.append(loss.cpu().data.numpy()[0])

		output_argmax = output.cpu().max(1)[1].data.numpy()

		output_labels.extend(output_argmax)
		target_labels.extend(target.cpu().data.numpy())

	return test_loss, temp_loss_epoch, output_labels, target_labels

def main():

	net = densenet.densenet201(sample_size=64, sample_duration=30, num_classes=num_classes)
	if pretrained:
		pretrained_weights = torch.load(pretrained_path)
		net.load_state_dict(pretrained_weights['state_dict'])

	train_dataset = MatrixDataset(base_dir='../data/', num_classes=num_classes)
	# train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)

	# test_dataset = MatrixDataset(base_dir='../data/test/', mode='test', sample_size=sample_size, num_classes=num_classes)
	# test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
	
	print len(train_dataset)
	# print len(test_dataset)

	train_loader, test_loader, class_weights = get_train_valid_loader(train_dataset, test_size=0.3)

	if sample_size:
		class_weights = np.ones((num_classes), dtype=np.float32)
		criterion = nn.CrossEntropyLoss()
	else:	
		class_weights = get_class_weights(train_dataset, train_loader)
		criterion = nn.CrossEntropyLoss(class_weights)
	
	optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)

	make_dir(weights_folder)
	make_dir(plot_folder)

	if use_cuda:
		net.cuda()

	### Plot Lists
	if resume == False:
		# for replacing whole plots
		train_loss = []
		train_loss_epoch = []
		precision_train = []
		recall_train = []
		avg_precision_train = []
		avg_recall_train = []
		f1_scores_train = []

		precision_test = []
		recall_test = []
		avg_precision_test = []
		avg_recall_test = []
		f1_scores_test = []
		test_loss = []
		test_loss_mean = []

	start_epoch = 0
	best_prec1 = 0
	early_stop_counter = 0

	### Resuming checkpoints
	if resume:
		checkpoint = torch.load(checkpoint_path)
		net.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		start_epoch = checkpoint['epoch'] # gives next epoch number
		best_prec1 = checkpoint['best_prec1']

	for epoch in xrange(start_epoch,epochs):

		### As I need shuffled dataset in every epoch!
		# train_dataset = AnatomyDataset(base_dir='../data/train/', mode='train', sample_size=sample_size, num_classes=num_classes)
		# train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
		
		### Plot lists
		if resume:
			# for appending to the plots
			train_loss = []
			train_loss_epoch = []
			precision_train = []
			recall_train = []
			avg_precision_train = []
			avg_recall_train = []
			f1_scores_train = []

			precision_test = []
			recall_test = []
			avg_precision_test = []
			avg_recall_test = []
			f1_scores_test = []
			test_loss = []
			test_loss_mean = []

		print "------------------------TRAINING {} - EPOCH: {}---------------------------".format(modelname,epoch)
		net.train()
		output_labels = []
		target_labels = []
		temp_loss_epoch = []

		train_loss, temp_loss_epoch, output_labels, target_labels = train(net, criterion, optimizer, epoch, train_loader, temp_loss_epoch, train_loss, output_labels, target_labels)

		### Accuracies
		train_loss_epoch.append(sum(temp_loss_epoch) / len(temp_loss_epoch))
		precision_train.append(np.asarray(precision_score(target_labels, output_labels, labels, average=None)))
		recall_train.append(np.asarray(recall_score(target_labels, output_labels, labels, average=None)))
		avg_precision_train.append(precision_score(target_labels, output_labels, labels, average='weighted'))
		avg_recall_train.append(recall_score(target_labels, output_labels, labels, average='weighted'))
		f1_scores_train.append(f1_score(target_labels, output_labels, labels, average='weighted'))
		
		### Printing
		print "precision_train ", precision_train[-1]
		print "recall_train ", recall_train[-1]
		print "train_loss_epoch", train_loss_epoch[-1]
		print "avg_precision_train ", avg_precision_train[-1]
		print "avg_recall_train ", avg_precision_train[-1]
		print "f1_scores_train ", f1_scores_train[-1]

		### Plotting
		plot_visdom(epoch, train_loss, win='train_loss')
		plot_visdom(epoch, train_loss_epoch, win='train_loss_epoch')
		plot_visdom(epoch, precision_train, win='precision_train')
		plot_visdom(epoch, recall_train, win='recall_train')
		plot_visdom(epoch, avg_precision_train, win='avg_precision_train')
		plot_visdom(epoch, avg_recall_train, win='avg_recall_train')
		plot_visdom(epoch, f1_scores_train, win='f1_scores_train')

		##### Testing ######

		print "------------------------TESTING EPOCH: {}---------------------------".format(epoch)
		net.eval()
		output_labels = []
		target_labels = []
		temp_loss_epoch = []

		test_loss, temp_loss_epoch, output_labels, target_labels = test(net, criterion, optimizer, epoch, test_loader, temp_loss_epoch, test_loss, output_labels, target_labels)

		### Accuracies
		test_loss_mean.append(sum(temp_loss_epoch) / len(temp_loss_epoch))
		precision_test.append(np.asarray(precision_score(target_labels, output_labels, labels, average=None)))
		recall_test.append(np.asarray(recall_score(target_labels, output_labels, labels, average=None)))
		avg_precision_test.append(precision_score(target_labels, output_labels, labels, average='weighted'))
		avg_recall_test.append(recall_score(target_labels, output_labels, labels, average='weighted'))
		f1_scores_test.append(f1_score(target_labels, output_labels, labels, average='weighted'))

		### Printing logs
		print "test loss mean ", test_loss_mean[-1]
		print "precision_test ", precision_test[-1]
		print "recall_test ", recall_test[-1]
		print "avg_precision_test ", avg_precision_test[-1]
		print "avg_recall_test ", avg_precision_test[-1]
		print "f1_scores_test ", f1_scores_test[-1]

		### Plotting
		epochm = epoch
		plot_visdom(epochm, test_loss, win='test_loss')
		plot_visdom(epochm, test_loss_mean, win='test_loss_mean')
		plot_visdom(epochm, precision_test, win='precision_test')
		plot_visdom(epochm, recall_test, win='recall_test')
		plot_visdom(epochm, avg_precision_test, win='avg_precision_test')
		plot_visdom(epochm, avg_recall_test, win='avg_recall_test')
		plot_visdom(epochm, f1_scores_test, win='f1_scores_test')

		### Save the environment
		vis.save(envs=[basename+modelname])

		### Checking if best
		# prec1 = test_loss_mean[-1]
		prec1 = f1_scores_test[-1]
		is_best = prec1 > best_prec1
		best_prec1 = max(prec1, best_prec1)
		### Saving weights
		if is_best or epoch % 10 == 0 or epoch==epochs-1:
			save_checkpoint({
				'epoch': epoch + 1,
				'arch': modelname,
				'state_dict': net.state_dict(),
				'best_prec1': best_prec1,
				'optimizer' : optimizer.state_dict(),
			}, is_best)

		# Early stopping # 
		# if is_best:
		# 	early_stop_counter = 0
		# else:
		# 	early_stop_counter += 1
		# if early_stop_counter > early_stopping:
		# 	print "Loss didn't fall for {} epochs, so doing an early stopping!".format(early_stopping)
		# 	break

		# ### Check accuracy to change loss weights
		# if epoch % 1 == 0 & epoch > 10:
		# 	class_weights[f1_scores_test[-1] <= min_acc_thresh] = 50.
		# 	print class_weights
		# 	criterion = nn.CrossEntropyLoss(torch.from_numpy(class_weights).cuda())

if __name__ == '__main__':
	main()