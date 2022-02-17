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

from demos.losses.synthesis import W_SSIM, SynthesisLoss

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
print(model_to_test.model)
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
    # imagePaths += glob("/home/abaozheng6/View_Synthesis/synsin/dataset/RealEstate10K/frames/train/000465ebe46a98d2/*.png")
    # imagePaths += glob("/home/abaozheng6/View_Synthesis/synsin/dataset/RealEstate10K/frames/train/00066b3649cc07e5/*.png")
    # imagePaths += glob("/home/abaozheng6/View_Synthesis/synsin/dataset/RealEstate10K/frames/train/0008631059fd7ba6/*.png")
    # imagePaths += glob("/home/abaozheng6/View_Synthesis/synsin/dataset/RealEstate10K/frames/train/000d73d2405332df/*.png")
    # imagePaths += glob("/home/abaozheng6/View_Synthesis/synsin/dataset/RealEstate10K/frames/train/00028da87cc5a4c4/*.png")
    # imagePaths += glob("/home/abaozheng6/View_Synthesis/synsin/dataset/RealEstate10K/frames/train/0006e8e3eaa8cd39/*.png")
    # imagePaths += glob("/home/abaozheng6/View_Synthesis/synsin/dataset/RealEstate10K/frames/train/00087de44e487f80/*.png")
    # imagePaths += glob("/home/abaozheng6/View_Synthesis/synsin/dataset/RealEstate10K/frames/train/000e285b03f3fddf/*.png")

    batch_size = 10
    status = []
    i = 0
    while i + batch_size <= len(imagePaths):

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

        i = j + 1

        t = time.perf_counter()
        # Generate a new view at the new transformation
        with torch.no_grad():
            pred_imgs = model_to_test.model.module.forward_angle(batch, RTS)
            # depth = nn.Sigmoid()(model_to_test.model.module.pts_regressor(batch['images'][0].cuda()))

        tmp = time.perf_counter() - t
        print(tmp)
        status.append(tmp)

    import numpy as np
    print(np.sum(status) / (len(status) * batch_size), len(status) * batch_size)


def testAcc():
    from glob import glob
    import time

    data_txt = [
        "0000cc6d8b108390",
        # "000465ebe46a98d2",
        # "00066b3649cc07e5",
        # "0008631059fd7ba6",
        # "000d73d2405332df",
        # "00028da87cc5a4c4",
        # "0006e8e3eaa8cd39",
        # "00087de44e487f80",
        # "000e285b03f3fddf"
    ]


    lossfcn = SynthesisLoss()

    for file_txt in data_txt:
        # imagePaths = glob("/home/abaozheng6/View_Synthesis/synsin/dataset/RealEstate10K/frames/train/{}/*.png".format(file_txt))
        frames = np.loadtxt("/home/abaozheng6/View_Synthesis/synsin/dataset/RealEstate10K/frames/train/{}.txt".format(file_txt), skiprows = 1)

        batch = {
            'images' : [],
            'cameras' : []
        }

        RTS = []

        PSNR_DATA = []
        W_PSNR_DATA = []
        SSIM_DATA = []
        W_SSIM_DATA = []

        for index in range(0, len(frames) - 1):

            frame = frames[index]
            
            imagePath = "/home/abaozheng6/View_Synthesis/synsin/dataset/RealEstate10K/frames/train/{}/{}.png".format(file_txt, str(int(frame[0])))
            im = Image.open(imagePath)
            im = transform(im)

            batch = {
                'images' : [im.unsqueeze(0)],
                'cameras' : [{
                    'K' : torch.eye(4).unsqueeze(0),
                    'Kinv' : torch.eye(4).unsqueeze(0)
                }]
            }

            ###############################################
            theta = 0#0.5 #-0.15
            phi = 0#-0.1
            tx = 1
            ty = 0
            tz = 0

            RT = torch.eye(4).unsqueeze(0)
            # Set up rotation
            RT[0,0:3,0:3] = torch.Tensor(quaternion.as_rotation_matrix(quaternion.from_rotation_vector([phi, theta, 0])))
            # Set up translation
            RT[0,0:3,3] = torch.Tensor([tx, ty, tz])
            # ALL RT
            RTS = [RT]
            print(RTS)
            ###############################################

            # intrinsics = frame[1:7]
            # extrinsics = frame[7:]

            # intrinsicMatrix = np.array(
            #     [
            #         [intrinsics[0], 0, intrinsics[2]],
            #         [0, intrinsics[1], intrinsics[3]],
            #         [0, 0, 1],
            #     ],
            #     dtype=np.float32
            # )

            # rotation = np.array([
            #     [extrinsics[0], extrinsics[1], extrinsics[2]],
            #     [extrinsics[4], extrinsics[5], extrinsics[6]],
            #     [extrinsics[10], extrinsics[9], extrinsics[10]]
            # ]).astype(np.float32)
            # translation = np.array([extrinsics[3], extrinsics[7], extrinsics[11]]).astype(np.float32)

            # rotation_inverse = np.linalg.inv(rotation)
            # translation_inverse = -rotation_inverse @ translation
            
            # if index == 0:
            #     extrinsicMatrix = np.array([
            #         [rotation[0, 0], rotation[0, 1], rotation[0, 2], translation[0]],
            #         [rotation[1, 0], rotation[1, 1], rotation[1, 2], translation[1]],
            #         [rotation[2, 0], rotation[2, 1], rotation[2, 2], translation[2]],
            #         [           0.0,            0.0,            0.0,            1.0]
            #     ]).astype(np.float32)

            # extrinsicMatrix_inverse = np.array([
            #     [rotation_inverse[0, 0], rotation_inverse[0, 1], rotation_inverse[0, 2], translation_inverse[0]],
            #     [rotation_inverse[1, 0], rotation_inverse[1, 1], rotation_inverse[1, 2], translation_inverse[1]],
            #     [rotation_inverse[2, 0], rotation_inverse[2, 1], rotation_inverse[2, 2], translation_inverse[2]],
            #     [                   0.0,                    0.0,                    0.0,                    1.0]
            # ]).astype(np.float32)

            # RTS = [torch.tensor([extrinsicMatrix_inverse * extrinsicMatrix])]
            # print(RTS)

            ###############################################

            # Generate a new view at the new transformation
            with torch.no_grad():
                pred_imgs = model_to_test.model.module.forward_angle(batch, RTS)
                # depth = nn.Sigmoid()(model_to_test.model.module.pts_regressor(batch['images'][0].cuda()))

            plt.imshow(im.permute(1,2,0) * 0.5 + 0.5)
            plt.savefig("/home/abaozheng6/View_Synthesis/synsin/demos/20220209/input/{}_{}_test_in.png".format(file_txt, str(int(frame[0]))))
            plt.imshow(pred_imgs[0].squeeze().cpu().permute(1,2,0).numpy() * 0.5 + 0.5)
            plt.savefig("/home/abaozheng6/View_Synthesis/synsin/demos/20220209/pred/{}_{}_test_pred.png".format(file_txt, str(int(frame[0]))))

            gtImagePath = "/home/abaozheng6/View_Synthesis/synsin/dataset/RealEstate10K/frames/train/{}/{}.png".format(file_txt, str(int(frames[0][0])))
            allLoss = lossfcn(pred_imgs[0].squeeze().cpu() * 0.5 + 0.5, transform(Image.open(gtImagePath)) * 0.5 + 0.5)

            PSNR_DATA.append(allLoss[0])
            SSIM_DATA.append(allLoss[1])
            W_PSNR_DATA.append(allLoss[2])
            W_SSIM_DATA.append(allLoss[3])

            if index == 30:
                break

            if index == 20:
                print(imagePath)
                break

        print("===========================")
        print("===========================")
        print("===========================")
        print(file_txt)
        print("PSNR_DATA = {}, W_PSNR_DATA = {}".format(np.mean(PSNR_DATA), np.mean(W_PSNR_DATA)))
        print("SSIM_DATA = {}, W_SSIM_DATA = {}".format(np.mean(SSIM_DATA), np.mean(W_SSIM_DATA)))
        print("===========================")
        print("===========================")
        print("===========================")


def simulation_test():
    from glob import glob
    import time

    batch = {
        'images' : [],
        'cameras' : []
    }

    lossfcn = SynthesisLoss()

    PSNR_DATA = []
    W_PSNR_DATA = []
    SSIM_DATA = []
    W_SSIM_DATA = []

    imagePaths = sorted(glob("/home/abaozheng6/View_Synthesis/synsin/demos/image_test_0217/*.png"))
    inputParameter = open("/home/abaozheng6/View_Synthesis/synsin/demos/image_test_0217/location.txt", "r").readlines()

    number = min(len(imagePaths), len(inputParameter))

    for index in range(number):

        imagePath = imagePaths[index]
        parameter = inputParameter[index]

        im = Image.open(imagePath)
        im = transform(im)

        batch = {
            'images' : [im.unsqueeze(0)],
            'cameras' : [{
                'K' : torch.eye(4).unsqueeze(0),
                'Kinv' : torch.eye(4).unsqueeze(0)
            }]
        }

        ###############################################

        theta, phi, tx, ty, tz = parameter.split(",")
        theta, phi, tx, ty, tz = float(theta) * np.pi / 180, float(phi) * np.pi / 180, float(tx), float(ty), float(tz)
        # theta, phi, tx, ty, tz = theta / 10, phi / 10, tx / 10, ty / 10, tz / 10
        theta, phi, tx, ty, tz = theta / 5, phi / 5, tx / 5, ty / 5, tz / 5
        # theta, phi, tx, ty, tz = theta, phi, tx / 10, ty / 10, tz / 10

        RT = torch.eye(4).unsqueeze(0)
        # Set up rotation
        RT[0,0:3,0:3] = torch.Tensor(quaternion.as_rotation_matrix(quaternion.from_rotation_vector([phi, theta, 0])))
        # Set up translation
        RT[0,0:3,3] = torch.Tensor([tx, ty, tz])
        # ALL RT
        RTS = [RT]

        # Generate a new view at the new transformation
        with torch.no_grad():
            pred_imgs = model_to_test.model.module.forward_angle(batch, RTS)
            # depth = nn.Sigmoid()(model_to_test.model.module.pts_regressor(batch['images'][0].cuda()))


        ###############################################

        theta, phi, tx, ty, tz = -theta, -phi, -tx, -ty, -tz

        RT = torch.eye(4).unsqueeze(0)
        # Set up rotation
        RT[0,0:3,0:3] = torch.Tensor(quaternion.as_rotation_matrix(quaternion.from_rotation_vector([phi, theta, 0])))
        # Set up translation
        RT[0,0:3,3] = torch.Tensor([tx, ty, tz])
        # ALL RT
        RTS = [RT]

        batch = {
            'images' : [pred_imgs[0].squeeze().unsqueeze(0)],
            'cameras' : [{
                'K' : torch.eye(4).unsqueeze(0),
                'Kinv' : torch.eye(4).unsqueeze(0)
            }]
        }

        # Generate a new view at the new transformation
        with torch.no_grad():
            pred_imgs = model_to_test.model.module.forward_angle(batch, RTS)
            # depth = nn.Sigmoid()(model_to_test.model.module.pts_regressor(batch['images'][0].cuda()))

        ###############################################
        # import cv2
        plt.axis("off")
        plt.imshow(im.permute(1,2,0) * 0.5 + 0.5)
        plt.savefig("/home/abaozheng6/View_Synthesis/synsin/demos/image_test_0217_result/input/test_in_{}.png".format(str(index).zfill(2)), bbox_inches = 'tight', pad_inches = 0)
        plt.axis("off")
        plt.imshow(pred_imgs[0].squeeze().cpu().permute(1,2,0).numpy() * 0.5 + 0.5)
        plt.savefig("/home/abaozheng6/View_Synthesis/synsin/demos/image_test_0217_result/pred/test_pred_{}.png".format(str(index).zfill(2)), bbox_inches = 'tight', pad_inches = 0)
        # cv2.imshow("/home/abaozheng6/View_Synthesis/synsin/demos/image_test_0217_result/pred/test_pred_{}.png".format(str(index).zfill(2)), pred_imgs[0].squeeze().cpu().permute(1,2,0).numpy() * 0.5 + 0.5)

        allLoss = lossfcn(pred_imgs[0].squeeze().cpu() * 0.5 + 0.5, im * 0.5 + 0.5)

        PSNR_DATA.append(allLoss[0])
        SSIM_DATA.append(allLoss[1])
        W_PSNR_DATA.append(allLoss[2])
        W_SSIM_DATA.append(allLoss[3])

    print("===========================")
    print("===========================")
    print("===========================")
    print("PSNR_DATA = {}, W_PSNR_DATA = {}".format(np.mean(PSNR_DATA), np.mean(W_PSNR_DATA)))
    print("SSIM_DATA = {}, W_SSIM_DATA = {}".format(np.mean(SSIM_DATA), np.mean(W_SSIM_DATA)))
    print("PSNR_DATA = {}, W_PSNR_DATA = {}".format(str(PSNR_DATA), str(W_PSNR_DATA)))
    print("SSIM_DATA = {}, W_SSIM_DATA = {}".format(str(SSIM_DATA), str(W_SSIM_DATA)))
    print("===========================")
    print("===========================")
    print("===========================")

if __name__ == "__main__":
    # testTime()
    # testAcc()
    simulation_test()
        




# In[ ]:




