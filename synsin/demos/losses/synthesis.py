# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn

from demos.losses.ssim import ssim, SSIM_Origin


class SynthesisLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_names = ("PSNR", "SSIM", "W_PSNR", "W_SSIM")
        self.type = ("psnr", "ssim", "psnr", "ssim")

    def get_loss_from_name(self, name):
        if name == "PSNR":
            loss = PSNR()
        elif name == "SSIM":
            loss = SSIM()
        elif name == "W_PSNR":
            loss = W_PSNR()
        elif name == "W_SSIM":
            loss = W_SSIM()

        if torch.cuda.is_available():
            return loss.cuda()
        else:
            return loss

    def forward(self, pred_img, gt_img):
        print(pred_img.size(), gt_img.size())
        ret = []
        for name, lossType in zip(self.loss_names, self.type):
            loss = self.get_loss_from_name(name)
            value = loss(pred_img, gt_img)
            print("{}: {}".format(name, value[lossType].data.cpu().numpy()))
            ret.append(value[lossType].data.cpu().numpy())
        return ret


import numpy as np
import copy
memo = {}
def getWeight(shape):
    m, n = shape[1], shape[2]
    if (m, n) not in memo:
        weight = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                weight[i][j] = np.cos((np.pi / n) * (j + 0.5 - n / 2))
        weight = np.array([[copy.deepcopy(weight), copy.deepcopy(weight), copy.deepcopy(weight)]])
        memo[m, n] = weight
    return memo[m, n]


class PSNR(nn.Module):
    def forward(self, pred_img, gt_img):
        pred_img = pred_img[None, :, :, :]
        gt_img = gt_img[None, :, :, :]

        # bs = pred_img.size(0)
        # mse_err = (pred_img - gt_img).pow(2).sum(dim=1).view(bs, -1).mean(dim=1)

        mse_err = ((pred_img - gt_img) ** 2).mean()

        psnr = 10 * (1 / mse_err).log10()
        return {"psnr": psnr}

class W_PSNR(nn.Module):
    def forward(self, pred_img, gt_img):
        weight = getWeight(pred_img.shape)
        pred_img = pred_img[None, :, :, :]
        gt_img = gt_img[None, :, :, :]
        weight = torch.tensor(weight)

        # bs = pred_img.size(0)
        # mse_err = ((pred_img - gt_img).pow(2) * weight).sum(dim=1).view(bs, -1).mean(dim=1) / weight.sum()
        # psnr = 10 * np.log10((1 / mse_err))

        mse_err = (((pred_img - gt_img) ** 2) * weight).sum() / weight.sum()
        psnr = 10 * (1 / mse_err).log10()
        
        return {"psnr": psnr}

class SSIM(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = SSIM_Origin()

    def forward(self, pred_img, gt_img):
        pred_img = pred_img[None, :, :, :]
        gt_img = gt_img[None, :, :, :]
        return {"ssim": self.loss(pred_img, gt_img)}

class W_SSIM(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = SSIM_Origin()

    def forward(self, pred_img, gt_img):
        weight = getWeight(pred_img.shape)
        weight = torch.tensor(weight)
        pred_img = pred_img[None, :, :, :]
        gt_img = gt_img[None, :, :, :]
        return {"ssim": self.loss(pred_img, gt_img, weight)}



