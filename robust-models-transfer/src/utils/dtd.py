from glob import glob 
from . import constants as cs
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from os.path import join as osj
from PIL import Image

class DTD(Dataset):
    def __init__(self, split="1", train=False):
        super().__init__()
        train_path = osj(cs.DTD_PATH, f"labels/train{split}.txt")
        val_path = osj(cs.DTD_PATH, f"labels/val{split}.txt")
        test_path = osj(cs.DTD_PATH, f"labels/test{split}.txt")
        if train:
            self.ims = open(train_path).readlines() + \
                            open(val_path).readlines()
        else:
            self.ims = open(test_path).readlines()
        
        self.full_ims = [osj(cs.DTD_PATH, "images", x) for x in self.ims]
        
        pth = osj(cs.DTD_PATH, f"labels/classes.txt")
        self.c_to_t = {x.strip(): i for i, x in enumerate(open(pth).readlines())}

        self.transform = cs.TRAIN_TRANSFORMS if train else \
                                         cs.TEST_TRANSFORMS
        self.labels = [self.c_to_t[x.split("/")[0]] for x in self.ims]

    def __getitem__(self, index):
        im = Image.open(self.full_ims[index].strip())
        im = self.transform(im)
        return im, self.labels[index]

    def __len__(self):
        return len(self.ims)

