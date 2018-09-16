import os
import sys
import time
import datetime
import argparse
import pickle
import csv
from skimage.data import imread
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from pdb import set_trace
sys.path.append('./PyTorch-YOLOv3_lite')

from matplotlib import pyplot as plt

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

host = "floyd"
if host is "local":
    args = ['--image_folder','../data_small/train',
            '--box_file','../data_small/box_data.pickle',
            '--model_config_path','./PyTorch-YOLOv3_lite/config/sholo.cfg',
            '--checkpoint_dir','../data_small/output',
            '--weights_path','../classifier_weights/6.weights',
            '--img_size','768',
            '--epochs','2',
            '--n_cpu','4',
            '--batch_size','2',
            '--use_cuda','True']
elif host is "floyd":       
    args = ['--image_folder','/airbus/train',
            '--box_file','/airbus/box_data.pickle',
            '--model_config_path','./PyTorch-YOLOv3_lite/config/sholo.cfg',
            '--checkpoint_dir','/output',
            '--weights_path','/weights/6.weights',
            '--img_size','768',
            '--epochs','100',
            '--n_cpu','4',
            '--batch_size','5',
            '--use_cuda','True']
else:
    raise NameError("invalid host name")

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--image_folder', type=str, default='data/samples', help='path to dataset')
parser.add_argument('--box_file', type=str, default='data/samples', help='path to box data')
parser.add_argument('--batch_size', type=int, default=16, help='size of each image batch')
parser.add_argument('--model_config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
parser.add_argument('--weights_path', type=str, default='weights/yolov3.weights', help='path to weights file')
parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
parser.add_argument('--checkpoint_interval', type=int, default=1, help='interval between saving model weights')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='directory where model checkpoints are saved')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
opt = parser.parse_args(args)
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

# Get hyper parameters
hyperparams     = parse_model_config(opt.model_config_path)[0]
learning_rate   = float(hyperparams['learning_rate'])
momentum        = float(hyperparams['momentum'])
decay           = float(hyperparams['decay'])
burn_in         = int(hyperparams['burn_in'])

# Initiate model
model = Darknet_Sholo(opt.model_config_path)
model.load_classifier_weights(opt.weights_path)
model.apply(weights_init_normal)

if cuda:
    model = model.cuda()

model.train()

# Get dataloader
with open(opt.box_file, 'rb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    data = pickle.load(f)
    
dataset = BoxDetections(opt.image_folder,opt.img_size,data)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, pin_memory=True)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=0, weight_decay=decay)
running_recall = np.zeros(100)
for epoch in range(opt.epochs):
    loss_list = []
    for batch_i, (img_file, imgs, targets) in enumerate(dataloader):
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)
        optimizer.zero_grad()

        loss = model(imgs, targets)

        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        running_recall[:-1] = running_recall[1:]
        running_recall[-1] = model.losses['recall']
        #print('[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, sin %f, cos %f, conf %f, cls %f, total %f, recall: %.5f]' %
        #                            (epoch, opt.epochs, batch_i, len(dataloader),
        #                            model.losses['x'], model.losses['y'], model.losses['w'],
        #                            model.losses['h'], model.losses['sin'],model.losses['cos'],model.losses['conf'], model.losses['cls'],
        #                           loss.item(), model.losses['recall']))

        print('{{"metric": "epoch", "value": {}}}'.format(epoch))
        print('{{"metric": "batch", "value": {}}}'.format(batch_i))
        print('{{"metric": "loss", "value": {}}}'.format(loss.item()))
        print('{{"metric": "log_loss", "value": {}}}'.format(np.log10(loss.item())))
        print('{{"metric": "loss_x", "value": {}}}'.format(model.losses['x']))
        print('{{"metric": "loss_y", "value": {}}}'.format(model.losses['y']))
        print('{{"metric": "loss_w", "value": {}}}'.format(model.losses['w']))
        print('{{"metric": "loss_h", "value": {}}}'.format(model.losses['h']))
        print('{{"metric": "loss_sin", "value": {}}}'.format(model.losses['sin']))
        print('{{"metric": "loss_cos", "value": {}}}'.format(model.losses['cos']))
        print('{{"metric": "loss_conf", "value": {}}}'.format(model.losses['conf']))
        print('{{"metric": "loss_class", "value": {}}}'.format(model.losses['cls']))
        print('{{"metric": "recall (running mean)", "value": {}}}'.format(np.mean(running_recall)))

        model.seen += imgs.size(0)

    if epoch % opt.checkpoint_interval == 0:
        model.save_weights('%s/%d.weights' % (opt.checkpoint_dir, epoch))
        with open('{}/loss.csv'.format(opt.checkpoint_dir), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow((epoch,np.mean(np.array(loss_list))))
