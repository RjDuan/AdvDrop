from Models.transformers import diet_tiny, diet_small, vit_tiny, vit_small
from torchvision import models
import torch.nn as nn
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import json
from utils import *
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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), ])

norm_layer = Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])


model = nn.Sequential(
    norm_layer,
    diet_tiny()
).to(device)

model = model.eval()
batch_size = 20
tar_cnt = 1000
cur_cnt = 0
suc_cnt = 0
eps = 4/255
steps = 50

data_dir = "./test_data"
normal_loader = get_data(data_dir="./test_data", transform=transform, batch_size=batch_size)
save_dir = "./results"
create_dir(save_dir)