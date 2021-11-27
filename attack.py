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
import torchattacks
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchattacks.attack import Attack
import argparse
from utils import *
from compression import *
from decompression import *
from PIL import ImageFile
import matplotlib

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        input = input / 255.0
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std
def save_img(img, img_name, save_dir):
    create_dir(save_dir)
    img_path = os.path.join(save_dir, img_name)
    img_pil = Image.fromarray(img.astype(np.uint8))
    img_pil.save(img_path)
def pred_label_and_confidence(model, input_batch, labels_to_class):
    input_batch = input_batch.cuda()
    with torch.no_grad():
        out = model(input_batch)
    _, index = torch.max(out, 1)

    percentage = torch.nn.functional.softmax(out, dim=1) * 100
    # print(percentage.shape)
    pred_list = []
    for i in range(index.shape[0]):
        pred_class = labels_to_class[index[i]]
        pred_conf = str(round(percentage[i][index[i]].item(), 2))
        pred_list.append([pred_class, pred_conf])
    return pred_list
ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), ])

norm_layer = Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
resnet_model = nn.Sequential(
    norm_layer,
    models.resnet50(pretrained=True)
).to(device)
resnet_model = resnet_model.eval()

attack_methods = {"PGD": [torchattacks.PGD, "l_inf"],
                "PGDL2": [torchattacks.PGDL2, "l2"],
                "FGSM": [torchattacks.FGSM, "l_inf"],
                "BIM": [torchattacks.BIM, "l_inf"],
                 "CW" : [torchattacks.CW, "l2"],
                 "Dfool": [torchattacks.DeepFool, "l2"]
                }

"""
We set epsilon = 4 for attacks on l∞ setting, set epsilon = 0.06 for attacks
on l2 setting that are common used in previous defense methods.
We set quantization table q with 100 for AdvDrop.
"""




batch_size = 10
tar_cnt = 1000
cur_cnt = 0
suc_cnt = 0
eps = 4/255
steps = 50


data_dir = "./test_data"
normal_loader = get_data(data_dir="./test_data", transform=transform, batch_size=batch_size)
save_dir = "./results"
create_dir(save_dir)

"""
eps: maximum perturbation
alpha: step_size
overshoot: parameter for enhancing the noise 
c : in the paper, parameter for box-constraint. (Default: 1e-4) 
kappa :  kappa (also written as ‘confidence’) in the paper. (Default: 0) 
lr (float) : learning rate of the Adam optimizer. (Default: 0.01)

>>>attack = torchattacks.DeepFool(model, steps=50, overshoot=0.02)
>>>adv_images = attack(images, labels)

>>> attack = torchattacks.PGDL2(model, eps=1.0, alpha=0.2, steps=40, random_start=True)
>>> adv_images = attack(images, labels)

>>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
>>> adv_images = attack(images, labels)


>>> attack = torchattacks.BIM(model, eps=4/255, alpha=1/255, steps=0)
>>> adv_images = attack(images, labels)

>>> attack = torchattacks.CW(model, c=1e-4, kappa=0, steps=1000, lr=0.01)
>>> adv_images = attack(images, labels)

>>> attack = torchattacks.FGSM(model, eps=0.007)
>>> adv_images = attack(images, labels)
default: untargetted attacks
"""
method = "PGD"
normal_iter = iter(normal_loader)
for i in range(tar_cnt // batch_size):
    print("Iter: ", i)
    images, labels = normal_iter.next()
    attack_method, bound = attack_methods[method][0], attack_methods[method][1]
    attack = attack_method(resnet_model, eps=eps, alpha=1 / 255, steps=steps,
                              random_start=True)
    at_images = attack(images, labels)
    outputs = resnet_model(at_images)
    _, pre = torch.max(outputs.data, 1)
    # Uncomment following codes if you wang to save the adv imgs
    at_images_np = at_images.detach().cpu().numpy()
    adv_img = at_images_np[0]
    adv_img = np.moveaxis(adv_img, 0, 2)
    adv_dir = os.path.join(save_dir, str(eps))
    create_dir(adv_dir)
    img_name = "adv_{}.jpg".format(i)
    adv_path = os.path.join(adv_dir, img_name)

    matplotlib.image.imsave(adv_path, adv_img)
    #save_img(adv_img, img_name, adv_dir)
    #labels = torch.from_numpy(np.random.randint(0, 1000, size=batch_size))



