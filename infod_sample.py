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

model_t =[models.resnet50]

q_sizes = [20, 60, 100]
model_names = ["resnet50_targetted"]
#model_names = ["diet_tiny_targetted", "diet_small_targetted", "vit_tiny_targetted", "vit_small_targetted"]
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
        return (input - mean.to(device=input.device)) / std.to(device=input.device)


    
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

"""
fool rate : Percentage of success of an adversarial attack, i.e., expected
number of times the attack is able to flip the label because of the
added perturbation.

"""
if __name__ == "__main__":
    idx_ = 0
    f = open("results.txt", "w")
    for next_model in model_t:

        print(f"{idx_}::: model_name: {model_names[idx_]}")
        name = model_names[idx_]
        
        for q_size in q_sizes:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            class_idx = json.load(open("./imagenet_class_index.json"))
            idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
            class2label = [class_idx[str(k)][0] for k in range(len(class_idx))]
            
            transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),]
            )

            norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # Data is normalized after dropping the information

            model = nn.Sequential(norm_layer,next_model(pretrained=True).to(device))
            model = model.eval()
            model_name = name
            batch_size = 20
            tar_cnt = 1000
            q_size = q_size
            cur_cnt = 0
            suc_cnt = 0
            data_dir = "./test-data"
            save_dir = "./results"
            data_clean(data_dir)
            normal_data = image_folder_custom_label(root=data_dir, transform=transform, idx2label=class2label)
            normal_loader = torch.utils.data.DataLoader(normal_data, batch_size=batch_size, shuffle=False)
            targetted_attack = True

            i =0
            fool_rate = 0
            file_number = 0
            for i, (images, labels) in enumerate(normal_loader): #in range(tar_cnt//batch_size):
                print("Iter: ", i)
                gt_labels = labels
                if targetted_attack:
                    labels = torch.from_numpy(np.random.randint(0, 1000, size= images.shape[0]))
                
                images = images * 255.0
                steps = 500 if targetted_attack else 50
                attack = InfoDrop(model, batch_size=images.shape[0], q_size =q_size, steps=steps, targeted= targetted_attack)
                at_images, at_labels, suc_step = attack(images, labels)
                ### Calculate fool rate
                outputs_pre_attack = model(images.to(device="cuda"))
                _, pred_pre_attack_label = torch.max(outputs_pre_attack.data, 1)
                fool_rate += torch.sum(pred_pre_attack_label!=at_labels)



                #Uncomment following codes if you wang to save the adv imgs
                at_images_np = at_images.detach().cpu().numpy()
                for idx, adv_img in enumerate(at_images_np):
                    adv_img = np.moveaxis(adv_img, 0, 2)
                    adv_dir = os.path.join(save_dir, f"{model_name}_{str(q_size)}")
                    create_dir(adv_dir)
                    create_dir(f"{adv_dir}/{class_idx[str(gt_labels[idx].item())][0]}")
                    img_name = f"{class_idx[str(gt_labels[idx].item())][0]}/adv_{file_number:04d}.jpg"
                    file_number += 1
                    save_img(adv_img, img_name, adv_dir)

                labels = labels.to(device)
                if targetted_attack:
                    suc_cnt += (at_labels == labels).sum().item()
                else:
                    suc_cnt += (at_labels != labels).sum().item()
                print("Current suc. rate: ", suc_cnt/((i+1)*batch_size))
            score_list = np.zeros(len(normal_data))
            score_list[:suc_cnt] = 1.0
            stderr_dist = np.std(np.array(score_list)) / np.sqrt(len(score_list))
            print('Avg suc rate: %.5f +/- %.5f' % (suc_cnt / len(normal_data), stderr_dist))
            print(f"Fool Rate {q_size} is : {fool_rate/len(normal_data)}")
            f.write(f"{name}_{q_size},{(suc_cnt / len(normal_data))}, {stderr_dist}, {fool_rate/len(normal_data)} \n")
        idx_ += 1

    f.close()