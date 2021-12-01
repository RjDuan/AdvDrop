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
from PIL import ImageFile
from info_attack import InfoDrop
from Models.transformers import diet_tiny, diet_small, vit_tiny, vit_small
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Normalize(nn.Module) :
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))
        
    def forward(self, input):
        # Broadcasting
        input = input/255.0
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
        pred_conf =  str(round(percentage[i][index[i]].item(),2))
        pred_list.append([pred_class, pred_conf])
    return pred_list

   
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_idx = json.load(open("./imagenet_class_index.json"))
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    class2label = [class_idx[str(k)][0] for k in range(len(class_idx))]
    
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),])  
    
    
    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    resnet_model = nn.Sequential(
        norm_layer,
        models.resnet50(pretrained=True)
    ).to(device)
    resnet_model = resnet_model.eval()
    
    # Uncomment if you want save results
    # save_dir = "./results"
    # create_dir(save_dir)
    batch_size = 20
    tar_cnt = 1000
    q_size = 40
    cur_cnt = 0
    suc_cnt = 0
    data_dir = "./test-data"
    data_clean(data_dir)
    normal_data = image_folder_custom_label(root=data_dir, transform=transform, idx2label=class2label)
    normal_loader = torch.utils.data.DataLoader(normal_data, batch_size=batch_size, shuffle=False)
    targetted_attack = False

    #normal_iter = iter(normal_loader)
    i =0
    for i, (images, labels) in enumerate(normal_loader): #in range(tar_cnt//batch_size):
        print("Iter: ", i)
        #images, labels = normal_iter.next()
        # For target attack: set random target. 
        # Comment if you set untargeted attack.
        if targetted_attack:
            labels = torch.from_numpy(np.random.randint(0, 1000, size = batch_size))
        
        images = images * 255.0
        steps = 500 if targetted_attack else 50
        attack = InfoDrop(resnet_model, batch_size=batch_size, q_size =q_size, steps=steps, targeted = targetted_attack)
        at_images, at_labels, suc_step = attack(images, labels)

        # Uncomment following codes if you wang to save the adv imgs
        # at_images_np = at_images.detach().cpu().numpy() 
        # adv_img = at_images_np[0]
        # adv_img = np.moveaxis(adv_img, 0, 2) 
        # adv_dir = os.path.join(save_dir, str(q_size))
        # img_name = "adv_{}.jpg".format(i)
        # save_img(adv_img, img_name, adv_dir)

        labels = labels.to(device)
        if targetted_attack:
            suc_cnt += (at_labels == labels).sum().item()
        else:
            suc_cnt += (at_labels != labels).sum().item()
        print("Current suc. rate: ", suc_cnt/((i+1)*batch_size))
    score_list = np.zeros(tar_cnt)
    score_list[:suc_cnt] = 1.0
    stderr_dist = np.std(np.array(score_list))/np.sqrt(len(score_list))
    print('Avg suc rate: %.5f +/- %.5f'%(suc_cnt/tar_cnt,stderr_dist))
