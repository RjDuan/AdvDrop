from Models.transformers import diet_tiny, diet_small, vit_tiny, vit_small
from torchvision import models
import torch.nn as nn
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import json
from utils import get_data, create_dir
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import ImageFile
from Models.transformers import vit_tiny, vit_small, diet_tiny, diet_small
ImageFile.LOAD_TRUNCATED_IMAGES = True

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_idx = json.load(open("./imagenet_class_index.json"))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
class2label = [class_idx[str(k)][0] for k in range(len(class_idx))]




OUT_DIR = '/tmp/'
#NUM_WORKERS = 16

from robustness import model_utils, datasets, train, defaults
from robustness.datasets import CIFAR, ImageNet
import torch
from cox.utils import Parameters
import cox.store
from torchvision import transforms
from torchvision import models

imagenet_ds = ImageNet('/tmp/')
# model , _ = model_utils.make_and_restore_model(arch='resnet50', dataset=imagenet_ds,
#                                                           resume_path="/home/hashmat.malik/PycharmProjects/Thesis/Robust Model Checkpoints/MadryLab/ResNet50/L2/imagenet_l2_3_0.pt", parallel=False)

## timm models
# model  = diet_small().to(device=device)
# config = resolve_data_config({}, model = model)
# transform = create_transform(**config)
### torch model
model = models.resnet50(pretrained=True).to(device=device)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])])



model = model.eval()
batch_size = 4

data_dir = "./test-data"
normal_loader = get_data(data_dir, transform, batch_size, class2label)


with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

correct = 0
total = 0
count = 0
for images, labels in normal_loader:
    print(count)
    count+=1
    images, labels = images.to(device), labels.to(device)
    preds = model(images)

    _, pre = torch.max(preds, 1)
    if torch.sum(labels==pre) < 2:
        print("Here")
    correct += torch.sum(pre==labels)
    total += images.shape[0]
print(f"The Model accuracy is {correct/total}")