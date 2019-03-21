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


parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--epochs', default=5, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')


def define_model(is_resnet, is_densenet, is_senet):
    use18 = True # True
    if is_resnet:
        if not use18:
            original_model = resnet.resnet18(pretrained = True)
            Encoder = modules.E_resnet(original_model) 
            model = net.model(Encoder, num_features=512, block_channel = [64, 128, 256, 512])
        else:
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
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model
   

def main():
    global args
    args = parser.parse_args()
    model = define_model(is_resnet=True, is_densenet=False, is_senet=False)
 
    if torch.cuda.device_count() == 8:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]).cuda()
        batch_size = 64
    elif torch.cuda.device_count() == 4:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
        batch_size = 32
    else:
        #model = model.cuda()
        batch_size = 4
        #batch_size = 11

    cudnn.benchmark = True
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    train_loader = loaddata.getTrainingData(batch_size)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_out_path = dir_path + '/model_output'
    model_out_path = Path(model_out_path)
    if not model_out_path.exists():
        model_out_path.mkdir()
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, optimizer, epoch)

        torch.save(model.state_dict(), model_out_path/ "model_epoch_{}.pth".format(epoch)) 
  #  save_checkpoint({'state_dict': model.state_dict()})


def train(train_loader, model, optimizer, epoch):
    criterion = nn.L1Loss()
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    cos = nn.CosineSimilarity(dim=1, eps=0)
    get_gradient = sobel.Sobel()#.cuda()

    end = time.time()
    for i, sample_batched in enumerate(train_loader):
        image, depth = sample_batched['image'], sample_batched['depth']

        #depth = depth.cuda(async=True)
        #depth = depth.cuda()
        #image = image.cuda()
        image = torch.autograd.Variable(image)
        depth = torch.autograd.Variable(depth)

        ones = torch.ones(depth.size(0), 1, depth.size(2),depth.size(3))#.float().cuda()
        ones = torch.autograd.Variable(ones)
        optimizer.zero_grad()

        output = model(image)
        output = torch.nn.functional.upsample(output, size=[depth.size(2),depth.size(3)], mode='bilinear')
        #print(image.size())
        #print(output.size())
        #gprint(depth.size())

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
    #lr = args.lr * (0.1 ** (epoch // 5))
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


if __name__ == '__main__':
    main()
