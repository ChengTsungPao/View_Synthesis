#!/usr/bin/env python
# coding: utf-8

# <h1> Demo Notebook</h1>
# 
# Notebook for visualising models on a given image.
# Given an image and a desired transformation, transform the image given the transformation.
# 

# In[1]:

# Cheng Fix
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 


# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import quaternion
import numpy as np

import os
os.chdir("..")
os.environ['DEBUG'] = '0'

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from models.networks.sync_batchnorm import convert_model
from models.base_model import BaseModel

from options.options import get_model

torch.backends.cudnn.enabled = True

MODEL_PATH = '/home/abaozheng6/View_Synthesis/synsin/modelcheckpoints/realestate/zbufferpts.pth'
BATCH_SIZE = 1

opts = torch.load(MODEL_PATH)['opts']
opts.render_ids = [1]

model = get_model(opts)

torch_devices = [int(gpu_id.strip()) for gpu_id in opts.gpu_ids.split(",")]
device = 'cuda:' + str(torch_devices[0])

if 'sync' in opts.norm_G:
    model = convert_model(model)
    model = nn.DataParallel(model, torch_devices[0:1]).cuda()
else:
    model = nn.DataParallel(model, torch_devices[0:1]).cuda()


#  Load the original model to be tested
model_to_test = BaseModel(model, opts)
model_to_test.load_state_dict(torch.load(MODEL_PATH)['state_dict'])
model_to_test.eval()


from PIL import Image

# Load the image
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

path = [
    "/home/abaozheng6/View_Synthesis/synsin/demos/im.jpg",
    "/home/abaozheng6/View_Synthesis/synsin/dataset/RealEstate10K/frames/train/0000cc6d8b108390/52553000.png",
    "/home/abaozheng6/View_Synthesis/synsin/dataset/RealEstate10K/frames/train/000465ebe46a98d2/240773867.png",
    "/home/abaozheng6/View_Synthesis/synsin/dataset/RealEstate10K/frames/train/00066b3649cc07e5/129195733.png",
    "/home/abaozheng6/View_Synthesis/synsin/dataset/RealEstate10K/frames/train/0008631059fd7ba6/170637133.png",
    "/home/abaozheng6/View_Synthesis/synsin/dataset/RealEstate10K/frames/train/000d73d2405332df/86686000.png",
    "/home/abaozheng6/View_Synthesis/synsin/dataset/RealEstate10K/frames/train/00028da87cc5a4c4/101601000.png",
    "/home/abaozheng6/View_Synthesis/synsin/dataset/RealEstate10K/frames/train/0006e8e3eaa8cd39/198431567.png",
    "/home/abaozheng6/View_Synthesis/synsin/dataset/RealEstate10K/frames/train/00087de44e487f80/100033283.png",
    "/home/abaozheng6/View_Synthesis/synsin/dataset/RealEstate10K/frames/train/000e285b03f3fddf/46980000.png"
]

def testTime():
    from glob import glob
    import time
    imagePaths = glob("/home/abaozheng6/View_Synthesis/synsin/dataset/RealEstate10K/frames/train/0000cc6d8b108390/*.png")

    batch_size = 10
    status = []
    i = 0
    while i + batch_size < len(imagePaths):

        batch = {
            'images' : [],
            'cameras' : []
        }

        RTS = []

        for j in range(i, i + batch_size):

            imagePath = imagePaths[j]
            
            im = Image.open(imagePath)
            im = transform(im)
            # Parameters for the transformation
            theta = -0.15
            phi = -0.1
            tx = 0
            ty = 0
            tz = 0.1

            RT = torch.eye(4).unsqueeze(0)
            # Set up rotation
            RT[0,0:3,0:3] = torch.Tensor(quaternion.as_rotation_matrix(quaternion.from_rotation_vector([phi, theta, 0])))
            # Set up translation
            RT[0,0:3,3] = torch.Tensor([tx, ty, tz])
            # ALL RT
            RTS += [RT]

            batch['images'] += [im.unsqueeze(0)]
            batch['cameras'] += [{
                    'K' : torch.eye(4).unsqueeze(0),
                    'Kinv' : torch.eye(4).unsqueeze(0)
            }]

        i = j

        t = time.perf_counter()
        # Generate a new view at the new transformation
        with torch.no_grad():
            pred_imgs = model_to_test.model.module.forward_angle(batch, RTS)
            # depth = nn.Sigmoid()(model_to_test.model.module.pts_regressor(batch['images'][0].cuda()))

        tmp = time.perf_counter() - t
        print(tmp)
        status.append(tmp)

    import numpy as np
    print(np.sum(status) / (len(status) * batch_size), len(status))

testTime()
        




# In[ ]:




