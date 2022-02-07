# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn

from demos.losses.ssim import ssim


class SynthesisLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_names = ("PSNR", "SSIM", "W_PSNR", "W_SSIM")


    def get_loss_from_name(self, name):
        if name == "l1":
            loss = L1LossWrapper()
        elif name == "PSNR":
            loss = PSNR()
        elif name == "SSIM":
            loss = SSIM()
        elif name == "W_PSNR":
            loss = W_PSNR()
        elif name == "W_SSIM":
            loss = W_SSIM()

        if torch.cuda.is_available():
            return loss.cuda()

    def forward(self, pred_img, gt_img):
        for name in self.loss_names:
            loss = self.get_loss_from_name(name)
            print("{}: {}".format(name, loss(pred_img, gt_img)))


import numpy as np
memo = {}
def getWeight(shape):
    m, n = shape
    if (m, n) not in memo:
        weight = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                weight[i][j] = np.cos((np.pi / n) * (j + 0.5 - n / 2))
        memo[m, n] = weight
    return memo[m, n]


class PSNR(nn.Module):
    def forward(self, pred_img, gt_img):
        bs = pred_img.size(0)
        mse_err = (pred_img - gt_img).pow(2).sum(dim=1).view(bs, -1).mean(dim=1)

        psnr = 10 * (1 / mse_err).log10()
        return {"psnr": psnr.mean()}

class W_PSNR(nn.Module):
    def forward(self, pred_img, gt_img):
        weight = getWeight(pred_img.shape)

        bs = pred_img.size(0)
        mse_err = ((pred_img - gt_img).pow(2) * weight).sum(dim=1).view(bs, -1).mean(dim=1) / np.sum(weight)

        psnr = 10 * (1 / mse_err).log10()
        return {"psnr": psnr.mean()}

class SSIM(nn.Module):
    def forward(self, pred_img, gt_img):
        return {"ssim": ssim(pred_img, gt_img)}

class W_SSIM(nn.Module):
    def forward(self, pred_img, gt_img):
        weight = getWeight(pred_img.shape)
        return {"ssim": ssim(pred_img, gt_img, weight)}

# Wrapper of the L1Loss so that the format matches what is expected
class L1LossWrapper(nn.Module):
    def forward(self, pred_img, gt_img):
        err = nn.L1Loss()(pred_img, gt_img)
        return {"L1": err, "Total Loss": err}

