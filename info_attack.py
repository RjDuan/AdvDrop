import numpy as np
import json
import os
import sys
import time
import math
import io
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from  torchattacks.attack import Attack
from utils import *
from compression import *
from decompression import *


class InfoDrop(Attack):
    r"""
    Distance Measure : l_inf bound on quantization table
    Arguments:
        model (nn.Module): model to attack.
        steps (int): number of steps. (DEFALUT: 40)
        batch_size (int): batch size
        q_size: bound for quantization table
        targeted: True for targeted attack
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    """

    def __init__(self, model, height=224, width=224, steps=40, batch_size=20, block_size=8, q_size=10, targeted=False):
        super(InfoDrop, self).__init__("InfoDrop", model)
        self.steps = steps
        self.targeted = targeted
        self.batch_size = batch_size
        self.height = height
        self.width = width
        # Value for quantization range
        self.factor_range = [5, q_size]
        # Differential quantization
        self.alpha_range = [0.1, 1e-20]
        self.alpha = torch.tensor(self.alpha_range[0])
        self.alpha_interval = torch.tensor((self.alpha_range[1] - self.alpha_range[0]) / self.steps)
        block_n = np.ceil(height / block_size) * np.ceil(height / block_size)
        q_ini_table = np.empty((batch_size, int(block_n), block_size, block_size), dtype=np.float32)
        q_ini_table.fill(q_size)
        self.q_tables = {"y": torch.from_numpy(q_ini_table),
                         "cb": torch.from_numpy(q_ini_table),
                         "cr": torch.from_numpy(q_ini_table)}

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        q_table = None
        self.alpha = self.alpha.to(self.device)
        self.alpha_interval = self.alpha_interval.to(self.device)

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam([self.q_tables["y"], self.q_tables["cb"], self.q_tables["cr"]], lr=0.01)

        images = images.permute(0, 2, 3, 1)
        components = {'y': images[:, :, :, 0], 'cb': images[:, :, :, 1], 'cr': images[:, :, :, 2]}
        for i in range(self.steps):
            self.q_tables["y"].requires_grad = True
            self.q_tables["cb"].requires_grad = True
            self.q_tables["cr"].requires_grad = True
            upresults = {}
            for k in components.keys():
                comp = block_splitting(components[k])
                comp = dct_8x8(comp)
                comp = quantize(comp, self.q_tables[k], self.alpha)
                comp = dequantize(comp, self.q_tables[k])
                comp = idct_8x8(comp)
                merge_comp = block_merging(comp, self.height, self.width)
                upresults[k] = merge_comp

            rgb_images = torch.cat(
                [upresults['y'].unsqueeze(3), upresults['cb'].unsqueeze(3), upresults['cr'].unsqueeze(3)], dim=3)
            rgb_images = rgb_images.permute(0, 3, 1, 2)
            outputs = self.model(rgb_images)
            _, pre = torch.max(outputs.data, 1)
            if self.targeted:
                suc_rate = ((pre == labels).sum() / self.batch_size).cpu().detach().numpy()
            else:
                suc_rate = ((pre != labels).sum() / self.batch_size).cpu().detach().numpy()

            adv_cost = adv_loss(outputs, labels)

            if not self.targeted:
                adv_cost = -1 * adv_cost

            total_cost = adv_cost
            optimizer.zero_grad()
            total_cost.backward()

            self.alpha += self.alpha_interval

            for k in self.q_tables.keys():
                self.q_tables[k] = self.q_tables[k].detach() - torch.sign(self.q_tables[k].grad)
                self.q_tables[k] = torch.clamp(self.q_tables[k], self.factor_range[0], self.factor_range[1]).detach()
            if i % 10 == 0:
                print('Step: ', i, "  Loss: ", total_cost.item(), "  Current Suc rate: ", suc_rate)
            if suc_rate >= 1:
                print('End at step {} with suc. rate {}'.format(i, suc_rate))
                q_images = torch.clamp(rgb_images, min=0, max=255.0).detach()
                return q_images, pre, i
        q_images = torch.clamp(rgb_images, min=0, max=255.0).detach()

        return q_images, pre, q_table
