from dataloader import DepthDatasetLoader
from torch.utils.data import Dataset, DataLoader
import torch
import logging
import sys
import os
import json
#from pathlib import Path
import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt
from skimage import io
#import torch
import torch.nn as nn
import torch.nn.functional as F
#import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from utilis import *
from model import GCN
from torch_cluster import knn_graph


torch.cuda.empty_cache()

dd_object = DepthDatasetLoader(dataset_directory = "data/")
# print(dd_object[10].keys())
# print(dd_object[10]['rgb'].shape)
#print(dd_object[192]['filenames'])
#dd_object[10]
print(len(dd_object))

batch_size = 1
n_val = 100
n_train = 400
# train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
train_set, val_set = random_split(dd_object, [n_train, n_val],generator=torch.Generator().manual_seed(0))
# 3. Create data loaders
loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# Declare Siamese Network
net = GCN()
net.load_state_dict(torch.load('/home/shida/pointCouldSegmentation/dir_checkpoint/checkpoint_epoch1.pth'))

#net.to(device=device)
print(net)


intrinsics_file = '/home/shida/pointCouldSegmentation/data/camera_intrinsic.json'
with open(intrinsics_file) as f:
    K = json.load(f)
K = np.array(K)
net.eval()
epoch_loss = 0
with torch.no_grad():
# with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
    for batch in val_loader:
        rgb1 = torch.squeeze(batch['rgb'],0).numpy()
        depth1 = torch.squeeze(batch['depth'],0).numpy()
        seg1 = torch.squeeze(batch['seg'],0).numpy()
        #seg_file = batch['filenames']
        #seg1 = read_rgb(seg_file)
        # rgb_img_file=batch['rgb_file']
        # depth_img_file=batch['depth_file']
        # seg_img_file=batch['seg_file']
        # print(rgb_img_file)
        # rgb1 = read_rgb(rgb_img_file)
        # depth1 = read_depth(depth_img_file)
        # seg1 = read_rgb(seg_img_file)
        #print(rgb1.size())

        # R_channel = seg1[:,:,0]
        # print(R_channel.max(),R_channel.min())
        # plt.imshow(R_channel)
        cityscapes_seg = labels_to_cityscapes_palette(seg1[:,:,0])
        pc1, color1 = depth_to_local_point_cloud(depth1, color=rgb1, k = K,max_depth=0.05)
        pc1, seg0islable = depth_to_local_point_cloud(depth1, color=seg1, k = K,max_depth=0.05)
        pc1.astype(int)
        #print('pc1:',pc1.dtype)
        #print('seg0:',seg0islable*255)

        
        x = torch.from_numpy(pc1)
        x=x.type(torch.float)
        edge_index = knn_graph(x, k=6)
        #print(edge_index.shape)
        #print(edge_index)
        #net.float()
        xx=x.to(device)
        edge_index=edge_index.to(device)
        out = net(x, edge_index)*255
        out = predictionlabels_to_cityscapes_palette(out.numpy())

        print(out[2673])
        cityscapes_seg = labels_to_cityscapes_palette(seg1[:,:,0])
        plt.imshow(cityscapes_seg)
        plt.show()
        img = point_cloud_to_image(pc1, out, K=K)
        plt.imshow(img)
        plt.show()
        break