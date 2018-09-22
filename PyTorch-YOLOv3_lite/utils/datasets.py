import glob
import random
import os
import numpy as np

import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.transform import resize
from skimage.data import imread
from skimage import draw
from skimage import transform 

import sys

class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.img_shape = (img_size, img_size)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image
        img = np.array(Image.open(img_path))
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        return img_path, input_img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 50

    def __getitem__(self, index):

        #---------
        #  Image
        #---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = np.array(Image.open(img_path))

        # Handles images with less than three channels
        while len(img.shape) != 3:
            index += 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path))

        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        #---------
        #  Label
        #---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            # Extract coordinates for unpadded + unscaled image
            x1 = w * (labels[:, 1] - labels[:, 3]/2)
            y1 = h * (labels[:, 2] - labels[:, 4]/2)
            x2 = w * (labels[:, 1] + labels[:, 3]/2)
            y2 = h * (labels[:, 2] + labels[:, 4]/2)
            # Adjust for added padding
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]
            # Calculate ratios from coordinates
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)

        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)

class BoxedDataset(Dataset):
    def __init__(self,imgdir,imgsize,data):
        super().__init__()
        self.imgdir = imgdir
        self.imgsize = imgsize
        self.box_image_ids = data['box_image_ids']
        self.L1 = data['L1']
        self.L2 = data['L2']
        self.R0 = data['C0'] # accidently switched when saved
        self.C0 = data['R0'] # accidently switched when saved
        self.theta = data['theta']
        self.all_image_ids = data['all_image_ids']
        self.num_ships = data['num_ships']
        self.total_ships = sum(self.num_ships)
        self.total_imgs = len(self.all_image_ids)
    def __len__(self):
        return 2*self.total_ships
    def __getitem__(self,idx):
        # actual ships
        label = 1 if (idx < self.total_ships) else 0
        if label == 1:
            Ri = self.R0[idx]
            Ci = self.C0[idx]
            Li = self.L1[idx]
            Rmax = self.imgsize[0]
            Cmax = self.imgsize[1]
            minrow = np.max([0,int(Ri-Li)])
            maxrow = np.min([Rmax,int(Ri+Li)])
            mincol = np.max([0,int(Ci-Li)])
            maxcol = np.min([Cmax,int(Ci+Li)])  
            img = imread(os.path.join(self.imgdir,self.box_image_ids[idx]))
            cropped_img = img[minrow:maxrow,mincol:maxcol,:]
            resized_img = transform.resize(cropped_img, (64,64), anti_aliasing=True)
        else:
            img_idx = np.random.choice(self.total_imgs)
            n = self.num_ships[img_idx]
            found_success = False
            tries = 0
            while not found_success:
                random_box_idx = np.random.choice(self.total_ships)
                Ri = self.R0[random_box_idx]
                Ci = self.C0[random_box_idx]
                Li = self.L1[random_box_idx]
                Rmax = self.imgsize[0]
                Cmax = self.imgsize[1]
                minrow = np.max([0,int(Ri-Li)])
                maxrow = np.min([Rmax,int(Ri+Li)])
                mincol = np.max([0,int(Ci-Li)])
                maxcol = np.min([Cmax,int(Ci+Li)]) 
                found_success = True
                actual_box_indices = [i for i, x in enumerate(self.box_image_ids) if x == self.all_image_ids[img_idx]]
                
                for actual_box_idx in actual_box_indices:
                    Rj = self.R0[actual_box_idx]
                    Cj = self.C0[actual_box_idx]
                    Lj = self.L1[actual_box_idx]
                    minrow_j = np.max([0,int(Rj-Lj)])
                    maxrow_j = np.min([Rmax,int(Rj+Lj)])
                    mincol_j = np.max([0,int(Cj-Lj)])
                    maxcol_j = np.min([Cmax,int(Cj+Lj)])                      
                    
                    # determine the (x, y)-coordinates of the intersection rectangle
                    xA = min(minrow, minrow_j)
                    yA = min(mincol, mincol_j)
                    xB = max(maxrow, maxrow_j)
                    yB = max(maxcol, maxcol_j)
                    
                    # compute the area of intersection rectangle
                    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                    if interArea > 0:
                        found_success = False
                tries += 1
                if tries > 10:
                    img_idx = np.random.choice(self.total_imgs)
                    n = self.num_ships[img_idx]
                    found_success = False
                    tries = 0
            img = imread(os.path.join(self.imgdir,self.all_image_ids[img_idx]))
            cropped_img = img[minrow:maxrow,mincol:maxcol,:]
            resized_img = transform.resize(cropped_img, (64,64), mode='reflect', anti_aliasing=True)                     
        return transforms.functional.to_tensor(resized_img), label

class BoxDetections(Dataset):
    def __init__(self,imgdir,imgsize,data):
        super().__init__()
        self.imgdir = imgdir
        self.imgsize = imgsize
        self.box_image_ids = data['box_image_ids']
        self.L1 = data['L1']
        self.L2 = data['L2']
        self.R0 = data['C0'] # accidently switched when saved
        self.C0 = data['R0'] # accidently switched when saved
        self.theta = data['theta']
        self.all_image_ids = data['all_image_ids']
        self.num_ships = data['num_ships']
        self.total_ships = sum(self.num_ships)
        self.total_imgs = len(self.all_image_ids)
        self.max_objects = 20
    def __len__(self):
        return self.total_imgs
    def __getitem__(self,idx):
        img_path = os.path.join(self.imgdir,self.all_image_ids[idx])
        img = np.array(Image.open(img_path)) 
        # Channels-first
        img = np.transpose(img, (2, 0, 1))
        # As pytorch tensor
        img = torch.from_numpy(img).float()
        # Find boxes in img
        # Fill matrix
        labels = np.zeros((self.max_objects, 6)) 
        if self.num_ships[idx] != 0:
            box_inds = [i for i,x in enumerate(self.box_image_ids) if x is self.all_image_ids[idx]]
            for i,ind in enumerate(box_inds[:self.max_objects]):
                L1 = self.L1[ind]
                L2 = self.L2[ind]
                R0 = self.R0[ind]
                C0 = self.C0[ind]
                theta = self.theta[ind]
                # Calculate ratios from coordinates
                labels[i, 1] = C0 
                labels[i, 2] = R0 
                labels[i, 3] = L1 
                labels[i, 4] = L2
                labels[i, 5] = theta
        labels = torch.from_numpy(labels)
        return img_path, img, labels
    

