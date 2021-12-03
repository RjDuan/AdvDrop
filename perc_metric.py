import lpips
import torchvision.transforms as transforms
import torch
from PIL import Image
import numpy as np
""" 
Compute perceptual similarity between the clean images and the adversarial images.
Compute it at different q-sizes. Find the size change in the adversarial images.

"""


loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

img_0_path = "/home/hashmat/Thesis/AdvDrop/results/0.01568627450980392/adv_0.jpg"
img_1_path = "/home/hashmat/Thesis/AdvDrop/test-data/n01440764/n01440764_ILSVRC2012_val_00021740.png"
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])
img0 = np.array(Image.open(img_0_path)) #torch.zeros(1,3,64,64) # image should be RGB, IMPORTANT: normalized to [-1,1]
img1 = np.array(Image.open(img_1_path)) #torch.zeros(1,3,64,64)
normalized_img0 = (img0 - np.amin(img0)) / (np.amax(img0) - np.amin(img0))
normalized_img1 = (img1 - np.amin(img1)) / (np.amax(img1) - np.amin(img1))
img0 = transform(normalized_img0).double()
img1 = transform(normalized_img1).double()
d = loss_fn_alex(img0, img1)
a=2