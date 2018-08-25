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
from pdb import set_trace as bp
sys.path.append('./PyTorch-YOLOv3_lite')

from matplotlib import pyplot as plt

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

args = ['--image_folder','/airbus/train',
        '--box_file','/airbus/box_data.pickle',
        '--model_config_path','./PyTorch-YOLOv3_lite/config/classifier.cfg',
        '--weights_path','./PyTorch-YOLOv3_lite/weights/classifier.weights',
        '--checkpoint_dir','/output',
        '--img_size','64',
        '--epochs','30',
        '--n_cpu','4',
        '--use_cuda','True']
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--image_folder', type=str, default='data/samples', help='path to dataset')
parser.add_argument('--box_file', type=str, default='data/samples', help='path to box data')
parser.add_argument('--batch_size', type=int, default=16, help='size of each image batch')
parser.add_argument('--model_config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
#parser.add_argument('--data_config_path', type=str, default='config/coco.data', help='path to data config file')
parser.add_argument('--weights_path', type=str, default='weights/yolov3.weights', help='path to weights file')
#parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')
#parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
#parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
parser.add_argument('--checkpoint_interval', type=int, default=1, help='interval between saving model weights')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='directory where model checkpoints are saved')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
opt = parser.parse_args(args)
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs('output', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

#classes = load_classes(opt.class_path)

# Get data configuration
#data_config     = parse_data_config(opt.data_config_path)
#train_path      = data_config['train']

# Get hyper parameters
hyperparams     = parse_model_config(opt.model_config_path)[0]
learning_rate   = float(hyperparams['learning_rate'])
momentum        = float(hyperparams['momentum'])
decay           = float(hyperparams['decay'])
burn_in         = int(hyperparams['burn_in'])

# Initiate model
model = Darknet_Classifier(opt.model_config_path)
#model.load_weights(opt.weights_path)
model.apply(weights_init_normal)

if cuda:
    model = model.cuda()

model.train()

# Get dataloader
with open(opt.box_file, 'rb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    data = pickle.load(f)
    
dataset = BoxedDataset(opt.image_folder,(768,768),data)
dataloader = torch.utils.data.DataLoader(
    dataset,batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu,pin_memory=True)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=0, weight_decay=decay)

for epoch in range(opt.epochs):
    loss_list = []
    for batch_i, (imgs, targets) in enumerate(dataloader):
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)

        optimizer.zero_grad()

        loss = model(imgs, targets)

        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        print('[Epoch %d/%d, Batch %d/%d] [Losses: %f]' %
                                    (epoch, opt.epochs-1, batch_i, len(dataloader),loss.item()))

        model.seen += imgs.size(0)
        
        if batch_i > 1000:
            break

    if epoch % opt.checkpoint_interval == 0:
        model.save_weights('%s/%d.weights' % (opt.checkpoint_dir, epoch))
        with open('{}/loss.csv'.format(opt.checkpoint_dir), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow((epoch,np.mean(np.array(loss_list))))
