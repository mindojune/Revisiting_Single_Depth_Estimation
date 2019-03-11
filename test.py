import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
from models import modules, net, resnet, densenet, senet
import loaddata
import util
import numpy as np
import sobel
from pathlib import Path
import os
from models_resnet import Resnet18_md, Resnet18Encoder

# TODO: instead of trying to update names, just use both versions of Resnet18 implemenation and load corresponding weights

def main():
    global is_resnet
    global is_densenet
    global is_senet
    global pretrain_logical
    is_resnet=True
    is_densenet= False
    is_senet= False
    pretrain_logical = False
    
    # cuda options
    # model = torch.nn.DataParallel(model).cuda()
    # model.load_state_dict(torch.load('./pretrained_model/model_senet'))
    #model = model.cuda().float()
    
    model = define_model()

    #model = Resnet18_md(3)  
    #model.load_state_dict(torch.load('/home/doragu/Dropbox/school/michigan/19w/3d-estimation-cnn/data/models/monodepth_resnet18_001.pth', map_location='cpu' ))
    
   
    #model.load_state_dict(torch.load('model_output/resnet_untrained/model_epoch_4.pth', map_location='cpu' ))
    #model.load_state_dict(torch.load('/home/doragu/Dropbox/school/michigan/19w/3d-estimation-cnn/data/models/monodepth_resnet18_001.pth', map_location='cpu' ))
    test_loader = loaddata.getTestingData(1)
    test(test_loader, model, 0.25)


def test(test_loader, model, thre):
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
            #depth = depth.cuda()
            #image = image.cuda()

            #image = torch.autograd.Variable(image, volatile=True)
            #depth = torch.autograd.Variable(depth, volatile=True)
            
            output = model(image)
            output = torch.nn.functional.upsample(output, size=[depth.size(2),depth.size(3)], mode='bilinear')
            #print(output.size())
            #print(depth.size())

            depth_edge = edge_detection(depth)
            output_edge = edge_detection(output)

            batchSize = depth.size(0)
            totalNumber = totalNumber + batchSize
            errors = util.evaluateError(output, depth)
            errorSum = util.addErrors(errorSum, errors, batchSize)
            averageError = util.averageErrors(errorSum, totalNumber)

            edge1_valid = (depth_edge > thre)
            edge2_valid = (output_edge > thre)

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


def define_model():
    if is_resnet:
        #original_model = resnet.resnet18(pretrained = pretrain_logical)
        #Encoder = modules.E_resnet(original_model) 
        #model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

        stereoModel = Resnet18Encoder(3)  
        model_dict = stereoModel.state_dict()
        encoder_dict = torch.load('/home/doragu/Dropbox/school/michigan/19w/3d-estimation-cnn/data/models/monodepth_resnet18_001.pth',map_location='cpu' )
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
    get_edge = sobel.Sobel()#.cuda()

    edge_xy = get_edge(depth)
    edge_sobel = torch.pow(edge_xy[:, 0, :, :], 2) + \
        torch.pow(edge_xy[:, 1, :, :], 2)
    edge_sobel = torch.sqrt(edge_sobel)

    return edge_sobel


if __name__ == '__main__':
    main()
