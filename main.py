import argparse

import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import loaddata
import util
import numpy as np
import sobel
from models import modules, net, resnet, densenet, senet
import os
from pathlib import Path

from models_resnet import Resnet18_md, Resnet18Encoder

import matplotlib.pyplot as plt
# 2019-03-19
# Coding TODO
# 1. program argument full specification (what options to allow) and implementation: 1V
# 2. write code so that if there is no such model file, grab it from url 
# 3. In training saving + continued training full specification and implementation
use_cuda = torch.cuda.is_available()

def define_test_model():
	#archs = {"Resnet", "Densenet", "SEnet", "Custom"}
	is_resnet = args.arch == "Resnet" #True #False #True
	is_densenet = args.arch == "Densenet" # #False #True #False # False
	is_senet = args.arch == "SEnet" # True #False #True #False
	is_custom = args.arch == "Custom"

	if is_resnet:
		#original_model = resnet.resnet18(pretrained = pretrain_logical)
		#Encoder = modules.E_resnet(original_model) 
		#model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

		stereoModel = Resnet18Encoder(3)  
		model_dict = stereoModel.state_dict()
		encoder_dict = torch.load('./models/monodepth_resnet18_001.pth',map_location='cpu' )
		new_dict = {}
		for key in encoder_dict:
			if key in model_dict:
				new_dict[key] = encoder_dict[key]

		stereoModel.load_state_dict(new_dict )
		
		Encoder = stereoModel

		model = net.model(Encoder, num_features=512, block_channel = [64, 128, 256, 512])

	if is_densenet:
		# TODO: no dot bug
		original_model = densenet.densenet161(pretrained=True)
		Encoder = modules.E_densenet(original_model)
		model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])

	if is_senet:
		original_model = senet.senet154(pretrained='imagenet')
		Encoder = modules.E_senet(original_model)
		model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

	return model   


def define_train_model():
	#archs = {"Resnet", "Densenet", "SEnet", "Custom"}
	is_resnet = args.arch == "Resnet" #True #False #True
	is_densenet = args.arch == "Densenet" # #False #True #False # False
	is_senet = args.arch == "SEnet" # True #False #True #False
	is_custom = args.arch == "Custom"

	use18 = True # True
	if is_resnet:
		if not use18:
			original_model = resnet.resnet18(pretrained = True)
			Encoder = modules.E_resnet(original_model) 
			model = net.model(Encoder, num_features=512, block_channel = [64, 128, 256, 512])
		else:
			stereoModel = Resnet18Encoder(3)  
			model_dict = stereoModel.state_dict()
			# 'pretrained_model/'
			encoder_dict = torch.load('./models/monodepth_resnet18_001.pth',map_location='cpu' )
			new_dict = {}
			for key in encoder_dict:
				if key in model_dict:
					new_dict[key] = encoder_dict[key]

			stereoModel.load_state_dict(new_dict )
			Encoder = stereoModel
			model = net.model(Encoder, num_features=512, block_channel = [64, 128, 256, 512])      
		  
	if is_densenet:
		original_model = densenet.densenet161(pretrained=True)
		Encoder = modules.E_densenet(original_model)
		model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
	if is_senet:
		original_model = senet.senet154(pretrained='imagenet')
		Encoder = modules.E_senet(original_model)
		model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

	return model

def edge_detection(depth):
	if use_cuda:
		get_edge = sobel.Sobel().cuda()
	else:
		get_edge = sobel.Sobel()

	edge_xy = get_edge(depth)
	edge_sobel = torch.pow(edge_xy[:, 0, :, :], 2) + \
		torch.pow(edge_xy[:, 1, :, :], 2)
	edge_sobel = torch.sqrt(edge_sobel)

	return edge_sobel

def visualize_image(image, b_output, output, depth):
	image = image.squeeze().permute(1,2,0)
	depth = depth.squeeze()
	b_output = b_output.squeeze()
	output = output.squeeze()
	errors = util.evaluateError(output, depth)
	
	titles = ["image", "b_prediction", "prediction", "gt"]
	images = [image, b_output, output, depth]
	cols = 1

	n_images = len(images)
	if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
	fig = plt.figure()
	for n, (image, title) in enumerate(zip(images, titles)):
		a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
		#if image.ndim == 2:
		#	plt.gray()
		plt.imshow(image)
		a.set_title(title)
	#plt.title(str(errors))
	print(str(errors))
	fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
	plt.show()
	

def test(thre):
	model = define_test_model()
	test_loader = loaddata.getTestingData(1)
	#test_loader = loaddata.getStyleTestingData(1)
	
	model.eval()

	totalNumber = 0

	Ae = 0
	Pe = 0
	Re = 0
	Fe = 0

	errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
				'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}

	with torch.no_grad():
		for i, sample_batched in enumerate(test_loader):
			image, depth = sample_batched['image'], sample_batched['depth']

			# depth = depth.cuda(async=True)

			if use_cuda:
				depth = depth.cuda()
				image = image.cuda()
			else:
				pass
				#image = torch.autograd.Variable(image, volatile=True)
				#depth = torch.autograd.Variable(depth, volatile=True)
			
			b_output = model(image)
			#output = torch.nn.functional.upsample(output, size=[depth.size(2),depth.size(3)], mode='bilinear')
			output = torch.nn.functional.interpolate(b_output, size=[depth.size(2),depth.size(3)], mode='bilinear', align_corners=False)
			visualize_image(image, b_output, output, depth)

			depth_edge = edge_detection(depth)
			output_edge = edge_detection(output)

			batchSize = depth.size(0)
			totalNumber = totalNumber + batchSize
			errors = util.evaluateError(output, depth)
			errorSum = util.addErrors(errorSum, errors, batchSize)
			averageError = util.averageErrors(errorSum, totalNumber)

			edge1_valid = (depth_edge > thre)
			edge2_valid = (output_edge > thre)
			#print(output_edge)
			#exit()

			nvalid = np.sum(torch.eq(edge1_valid, edge2_valid).float().data.cpu().numpy())
			A = nvalid / (depth.size(2)*depth.size(3))

			nvalid2 = np.sum(((edge1_valid + edge2_valid) ==2).float().data.cpu().numpy())
			P = nvalid2 / (np.sum(edge2_valid.data.cpu().numpy()))
			R = nvalid2 / (np.sum(edge1_valid.data.cpu().numpy()))

			F = (2 * P * R) / (P + R)

			Ae += A
			Pe += P
			Re += R
			Fe += F
			print('Epoch: [{0}/{1}]\t' .format( i, len(test_loader)))    
			

	Av = Ae / totalNumber
	Pv = Pe / totalNumber
	Rv = Re / totalNumber
	Fv = Fe / totalNumber
	print('PV', Pv)
	print('RV', Rv)
	print('FV', Fv)

	averageError['RMSE'] = np.sqrt(averageError['MSE'])
	print(averageError)

	if is_resnet:
	   if pretrain_logical: 
		   save_name = 'resnet_pretrained'
	   else:
		   save_name = 'renet_untrained'
	elif is_densenet:
	   if pretrain_logical:
		   save_name = 'densenet_pretrained'
	   else:
		   save_name = 'densenet_untrained'
	else:
	   if pretrain_logical:
		   save_name = 'senet_pretrained'
	   else:
		   save_name = 'senet_untrained'

	dir_path = os.path.dirname(os.path.realpath(__file__))
	result_out_path = Path(dir_path +'/csvs')
	if not result_out_path.exists():
		result_out_path.mkdir()

	with open('csvs/'+save_name+'.csv', 'w') as sub:
		sub.write('RV' + str(Rv) + '\n')
		sub.write('FV' + str(Fv) + '\n')
		sub.write('RMSE'+ str(averageError['RMSE'])  + '\n')
	print('Done!') 
	return

def train(train_loader, model, optimizer, epoch):
	criterion = nn.L1Loss()
	batch_time = AverageMeter()
	losses = AverageMeter()

	model.train()

	cos = nn.CosineSimilarity(dim=1, eps=0)
	if use_cuda:
		get_gradient = sobel.Sobel().cuda()
	else:
		get_gradient = sobel.Sobel()#.cuda()

	end = time.time()
	for i, sample_batched in enumerate(train_loader):
		image, depth = sample_batched['image'], sample_batched['depth']

		#depth = depth.cuda(async=True)
		if use_cuda:
			depth = depth.cuda()
			image = image.cuda()
		else:
			image = torch.autograd.Variable(image)
			depth = torch.autograd.Variable(depth)

		ones = torch.ones(depth.size(0), 1, depth.size(2),depth.size(3)).float().cuda()
		ones = torch.autograd.Variable(ones)
		optimizer.zero_grad()

		output = model(image)
		#output = torch.nn.functional.upsample(output, size=[depth.size(2),depth.size(3)], mode='bilinear')
		output = torch.nn.functional.interpolate(output, size=[depth.size(2),depth.size(3)], mode='bilinear', align_corners=False)

		depth_grad = get_gradient(depth)
		output_grad = get_gradient(output)
		depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
		depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
		output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
		output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)

		depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
		output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

		# depth_normal = F.normalize(depth_normal, p=2, dim=1)
		# output_normal = F.normalize(output_normal, p=2, dim=1)

		loss_depth = torch.log(torch.abs(output - depth) + 0.5).mean()
		loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
		loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()
		loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()
		
		# TODO: grad_dx, grad_dy being negative: is it ok or is something wrong here?
		#print("losses:",loss_depth, loss_dx, loss_dy, loss_normal)
		loss = loss_depth + loss_normal + (loss_dx + loss_dy)

		#losses.update(loss.data[0], image.size(0))
		losses.update(loss.data.item(), image.size(0))
		loss.backward()
		optimizer.step()

		batch_time.update(time.time() - end)
		end = time.time()
   
		batchSize = depth.size(0)

		print('Epoch: [{0}][{1}/{2}]\t'
		  'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
		  'Loss {loss.val:.4f} ({loss.avg:.4f})'
		  .format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses))
 

def adjust_learning_rate(optimizer, epoch):
	lr = args.lr * (0.3 ** (epoch // 5))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def save_checkpoint(state, filename='checkpoint.pth.tar'):
	filename = 'resnet_untrained.pth'
	torch.save(state, filename)


def main():
	parser = argparse.ArgumentParser(description='PyTorch Depth Estimation')
	parser.add_argument('--mode', default="test", type=str, help='number of total epochs to run')

	parser.add_argument('--premodel', default="scratch", 
						type=str, help='pretrained model options: imagenet, stereo_view, scratch')

	# conflict with mode argument
	# parser.add_argument('--finetune', default=False, 
	# 					type=bool,
	# 					help='pretrained model options: imagenet, stereo_view, scratch')
	
	# doing this will result in ignoring the premodel argument
	parser.add_argument('--model', default="None", #default="./models/monodepth_resnet18_001.pth", 
						type=str, help='filepath of the model')
	parser.add_argument('--arch', default="Resnet", #default="./models/monodepth_resnet18_001.pth", 
						type=str, help='choice of architecture')
	parser.add_argument('--epochs', default=5, type=int,
						help='number of total epochs to run')
	parser.add_argument('--start-epoch', default=0, type=int,
						help='manual epoch number (useful on restarts)')
	parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
						help='initial learning rate')
	parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
	parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
						help='weight decay (default: 1e-4)')
		
	global args
	args = parser.parse_args()
	modes = {"test", "train"}
	premodels = {"imagenet", "stereo_view", "scratch"}
	archs = {"Resnet", "Densenet", "SEnet", "Custom"}
	
	if args.mode not in modes or args.premodel not in premodels or args.arch not in archs:
		print("invalid arguments!")
		exit(1)

	if args.mode == "test":
		threshold = 0.25
		test(threshold)
	else:
		model = define_train_model()
 
		if torch.cuda.device_count() == 8:
			model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]).cuda()
			batch_size = 64
		elif torch.cuda.device_count() == 4:
			model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
			batch_size = 32
		else:
			model = model.cuda()
			batch_size = 16
			#batch_size = 11

		cudnn.benchmark = True
		optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

		train_loader = loaddata.getTrainingData(batch_size)
		#train_loader = loaddata.getStyleTrainingData(batch_size)
		dir_path = os.path.dirname(os.path.realpath(__file__))
		model_out_path = dir_path + '/model_output'
		model_out_path = Path(model_out_path)
		if not model_out_path.exists():
			model_out_path.mkdir()
		for epoch in range(args.start_epoch, args.epochs):
			adjust_learning_rate(optimizer, epoch)
			train(train_loader, model, optimizer, epoch)

			torch.save(model.state_dict(), model_out_path/ model.arch+"_model_epoch_{}.pth".format(epoch)) 


if __name__ == '__main__':
	main()
