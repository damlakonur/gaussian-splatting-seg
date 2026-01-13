#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def tensor2image(tensor, normalized=True):
    if normalized:
        tensor = tensor.clamp(0, 1)
        image = tensor.permute(1, 2, 0).detach().cpu().numpy()
        image = (image * 255).astype(np.uint8)
    else:
        tensor = tensor.clamp(0, 255)
        image = tensor.permute(1, 2, 0).detach().cpu().numpy()
        image = (image).astype(np.uint8)
    return image
