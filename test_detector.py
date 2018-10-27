from __future__ import division
import os, copy
import matplotlib.pyplot as plt
from skimage import draw
import sys
import time
import datetime
import argparse
#from tqdm import tqdm
import pdb
import pickle
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
sys.path.append('./PyTorch-YOLOv3_oglite')

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *


host = "local"
if host is "local":
    args = ['--image_folder','../data_bbox_only/train',
            '--box_file','../data_bbox_only/box_data.pickle',
            '--model_config_path','./PyTorch-YOLOv3_oglite/config/yolov3.cfg',
            '--weights_path','../detector_weights/mschwendeman-projects-airbus_ship_detection-88-output/9.weights',
            '--img_size','768',
            '--epochs','2',
            '--n_cpu','4',
            '--batch_size','1',
            '--use_cuda','True',
            '--conf_thres','0.5',
            '--nms_thres','0.45']
elif host is "floyd":       
    args = ['--image_folder','/airbus/train',
            '--box_file','/airbus/box_data.pickle',
            '--model_config_path','./PyTorch-YOLOv3_lite/config/sholo.cfg',
            '--weights_path','/weights/35.weights.txt',
            '--img_size','768',
            '--epochs','100',
            '--n_cpu','4',
            '--batch_size','5',
            '--use_cuda','True',
            '--conf_thres','0.01']
else:
    raise NameError("invalid host name")

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--image_folder', type=str, default='data/samples', help='path to dataset')
parser.add_argument('--box_file', type=str, default='data/samples', help='path to box data')
parser.add_argument('--batch_size', type=int, default=16, help='size of each image batch')
parser.add_argument('--model_config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
parser.add_argument('--weights_path', type=str, default='weights/yolov3.weights', help='path to weights file')
parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')
parser.add_argument('--iou_thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
parser.add_argument('--conf_thres', type=float, default=0.5, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.45, help='iou thresshold for non-maximum suppression')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
opt = parser.parse_args(args)
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda

# Get data configuration
test_path       = opt.image_folder
num_classes     = 0

# Initiate model
model = Darknet(opt.model_config_path)
model.load_weights(opt.weights_path)
if cuda:
    model = model.cuda()

model.eval()

# Get dataloader
with open(opt.box_file, 'rb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    data = pickle.load(f)
    
dataset = BoxDetections(opt.image_folder,opt.img_size,data)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, pin_memory=True)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

n_gt = 0
correct = 0

print ('Compute mAP...')

outputs = []
targets = None
APs = []
for batch_i, (img_file, imgs, targets) in enumerate(dataloader):
    imgs = Variable(imgs.type(Tensor))
    targets = targets.type(Tensor)

    with torch.no_grad():
        output_1 = model(imgs, targets)
        #pdb.set_trace()
        output_2 = model(imgs)
        output = non_max_suppression(output_2, 1, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)

    # Compute average precision for each sample
    for sample_i in range(targets.size(0)):
        correct = []

        # Get labels for sample where width is not zero (dummies)
        #annotations = targets[sample_i, targets[sample_i, :, 3] != 0]
        annotations = targets[sample_i]
        # Extract detections
        detections = output[sample_i]
        if detections is None:
            # If there are no detections but there are annotations mask as zero AP
            if annotations.size(0) != 0:
                APs.append(0)
            continue

        # Get detections sorted by decreasing confidence scores
        detections = detections[np.argsort(-detections[:, 4])]

        # If no annotations add number of detections as incorrect
        if annotations.size(0) == 0:
            correct.extend([0 for _ in range(len(detections))])
        else:
            # Extract target boxes as (x1, y1, x2, y2)
            #target_boxes = torch.FloatTensor(annotations[:, 1:])
            target_boxes = torch.FloatTensor(annotations[:, 1:].shape)
            target_boxes[:, 0] = (annotations[:, 1] - annotations[:, 3] / 2)
            target_boxes[:, 1] = (annotations[:, 2] - annotations[:, 4] / 2)
            target_boxes[:, 2] = (annotations[:, 1] + annotations[:, 3] / 2)
            target_boxes[:, 3] = (annotations[:, 2] + annotations[:, 4] / 2)
            target_boxes *= opt.img_size
            detected = []
            
            im = imgs[sample_i].cpu().permute([1,2,0]).numpy().astype('int')
            im2 = copy.deepcopy(im)
            im3 = copy.deepcopy(im)
            #pdb.set_trace()
            for box in target_boxes:
                #pdb.set_trace()
                #rr, cc = draw.polygon_perimeter([box[1], box[3], box[3], box[1]],[box[0], box[0], box[2], box[2]], shape=im.shape)
                rr, cc = draw.polygon([box[1], box[3], box[3], box[1]],[box[0], box[0], box[2], box[2]], shape=im.shape)
                im2[rr,cc,0] = 255
                im2[rr,cc,1] = 0
                im2[rr,cc,2] = 0

            #for *pred_bbox, conf, obj_conf, obj_pred in detections:
            for *pred_bbox, conf in detections:

                pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1)
                # Compute iou with target boxes
                iou = bbox_iou(pred_bbox, target_boxes, x1y1x2y2=True)
                #print(iou)
                # Extract index of largest overlap
                best_i = np.argmax(iou)
                # If overlap exceeds threshold and classification is correct mark as correct
                #if iou[best_i] > opt.iou_thres and obj_pred == annotations[best_i, 0] and best_i not in detected:
                if iou[best_i] > opt.iou_thres and best_i not in detected:
                    correct.append(1)
                    detected.append(best_i)
                else:
                    correct.append(0)
                box = pred_bbox[0]
                #pdb.set_trace()
                rr, cc = draw.polygon([box[1], box[3], box[3], box[1]],[box[0], box[0], box[2], box[2]], shape=im.shape)
                im3[rr,cc,0] = 0
                im3[rr,cc,1] = 255
                im3[rr,cc,2] = 0

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(im, vmin=0, vmax=255)
        ax[0].axis('off')
        ax[1].imshow(im2, vmin=0, vmax=255)
        ax[1].axis('off')
        ax[2].imshow(im3, vmin=0, vmax=255)
        ax[2].axis('off')
        fig.show()
        #pdb.set_trace()
        
        # Extract true and false positives
        true_positives  = np.array(correct)
        false_positives = 1 - true_positives

        # Compute cumulative false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # Compute recall and precision at all ranks
        recall    = true_positives / annotations.size(0) if annotations.size(0) else true_positives
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # Compute average precision
        AP = compute_ap(recall, precision)
        APs.append(AP)

        print ("+ Sample [%d/%d] AP: %.4f (%.4f)" % (len(APs), len(dataset), AP, np.mean(APs)))

print("Mean Average Precision: %.4f" % np.mean(APs))
