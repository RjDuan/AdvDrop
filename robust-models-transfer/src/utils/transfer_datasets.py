from robustness.datasets import DataSet, CIFAR
from robustness import data_augmentation as da
import torch as ch
from . import constants as cs
from torchvision.datasets import CIFAR100
from .caltech import Caltech101, Caltech256

from . import aircraft, food_101, dtd
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

class ImageNetTransfer(DataSet):
    def __init__(self, data_path, **kwargs):
        ds_kwargs = {
            'num_classes': kwargs['num_classes'],
            'mean': ch.tensor(kwargs['mean']),
            'custom_class': None,
            'std': ch.tensor(kwargs['std']),
            'transform_train': cs.TRAIN_TRANSFORMS,
            'label_mapping': None,
            'transform_test': cs.TEST_TRANSFORMS
        }
        super(ImageNetTransfer, self).__init__(kwargs['name'], data_path, **ds_kwargs)

class TransformedDataset(Dataset):
    def __init__(self, ds, transform=None):
        self.transform = transform
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample, label = self.ds[idx]
        if self.transform:
            sample = self.transform(sample)
            if sample.shape[0] == 1:
                sample = sample.repeat(3,1,1)
        return sample, label

def make_loaders_pets(batch_size, workers):
    ds = ImageNetTransfer(cs.PETS_PATH, num_classes=37, name='pets',
                            mean=[0., 0., 0.], std=[1., 1., 1.])
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_birds(batch_size, workers):
    ds = ImageNetTransfer(cs.BIRDS_PATH, num_classes=500, name='birds',
                            mean=[0.,0.,0.], std=[1.,1.,1.])
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_SUN(batch_size, workers):
    ds = ImageNetTransfer(cs.SUN_PATH, num_classes=397, name='SUN397',
                            mean=[0.,0.,0.], std=[1.,1.,1.])
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_CIFAR10(batch_size, workers, subset):
    ds = CIFAR('/tmp')
    ds.transform_train = cs.TRAIN_TRANSFORMS
    ds.transform_test = cs.TEST_TRANSFORMS
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers, subset=subset)

def make_loaders_CIFAR100(batch_size, workers, subset):
    ds = ImageNetTransfer('/tmp', num_classes=100, name='cifar100',
                    mean=[0.5071, 0.4867, 0.4408],
                    std=[0.2675, 0.2565, 0.2761])
    ds.custom_class = CIFAR100
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers, subset=subset)

def make_loaders_oxford(batch_size, workers):
    ds = ImageNetTransfer(cs.FLOWERS_PATH, num_classes=102,
            name='oxford_flowers', mean=[0.,0.,0.],
            std=[1.,1.,1.])
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_aircraft(batch_size, workers):
    ds = ImageNetTransfer(cs.FGVC_PATH, num_classes=100, name='aircraft',
                    mean=[0.,0.,0.], std=[1.,1.,1.])
    ds.custom_class = aircraft.FGVCAircraft
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_food(batch_size, workers):
    food = food_101.FOOD101()
    train_ds, valid_ds, classes =  food.get_dataset()
    train_dl, valid_dl = food.get_dls(train_ds, valid_ds, bs=batch_size,
                                                    num_workers=workers)
    return 101, (train_dl, valid_dl)

def make_loaders_caltech101(batch_size, workers):
    ds = Caltech101(cs.CALTECH101_PATH, download=True)
    np.random.seed(0)
    ch.manual_seed(0)
    ch.cuda.manual_seed(0)
    ch.cuda.manual_seed_all(0)
    NUM_TRAINING_SAMPLES_PER_CLASS = 30

    class_start_idx = [0]+ [i for i in np.arange(1, len(ds)) if ds.y[i]==ds.y[i-1]+1]

    train_indices = sum([np.arange(start_idx,start_idx + NUM_TRAINING_SAMPLES_PER_CLASS).tolist() for start_idx in class_start_idx],[])
    test_indices = list((set(np.arange(1, len(ds))) - set(train_indices) ))

    train_set = Subset(ds, train_indices)
    test_set = Subset(ds, test_indices)

    train_set = TransformedDataset(train_set, transform=cs.TRAIN_TRANSFORMS) 
    test_set = TransformedDataset(test_set, transform=cs.TEST_TRANSFORMS)

    return 101, [DataLoader(d, batch_size=batch_size, shuffle=True,
                num_workers=workers) for d in (train_set, test_set)]


def make_loaders_caltech256(batch_size, workers):
    ds = Caltech256(cs.CALTECH256_PATH, download=True)
    np.random.seed(0)
    ch.manual_seed(0)
    ch.cuda.manual_seed(0)
    ch.cuda.manual_seed_all(0)
    NUM_TRAINING_SAMPLES_PER_CLASS = 60

    class_start_idx = [0]+ [i for i in np.arange(1, len(ds)) if ds.y[i]==ds.y[i-1]+1]

    train_indices = sum([np.arange(start_idx,start_idx + NUM_TRAINING_SAMPLES_PER_CLASS).tolist() for start_idx in class_start_idx],[])
    test_indices = list((set(np.arange(1, len(ds))) - set(train_indices) ))

    train_set = Subset(ds, train_indices)
    test_set = Subset(ds, test_indices)

    train_set = TransformedDataset(train_set, transform=cs.TRAIN_TRANSFORMS) 
    test_set = TransformedDataset(test_set, transform=cs.TEST_TRANSFORMS)

    return 257, [DataLoader(d, batch_size=batch_size, shuffle=True,
                num_workers=workers) for d in (train_set, test_set)]


def make_loaders_dtd(batch_size, workers):
        train_set = dtd.DTD(train=True)
        val_set = dtd.DTD(train=False)
        return 57, [DataLoader(ds, batch_size=batch_size, shuffle=True,
                num_workers=workers) for ds in (train_set, val_set)]

def make_loaders_cars(batch_size, workers):
    ds = ImageNetTransfer(cs.CARS_PATH, num_classes=196, name='stanford_cars',
                            mean=[0.,0.,0.], std=[1.,1.,1.])
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

DS_TO_FUNC = {
    "dtd": make_loaders_dtd,
    "stanford_cars": make_loaders_cars,
    "cifar10": make_loaders_CIFAR10,
    "cifar100": make_loaders_CIFAR100,
    "SUN397": make_loaders_SUN,
    "aircraft": make_loaders_aircraft,
    "flowers": make_loaders_oxford,
    "food": make_loaders_food,
    "birds": make_loaders_birds,
    "caltech101": make_loaders_caltech101,
    "caltech256": make_loaders_caltech256,
    "pets": make_loaders_pets,
}

def make_loaders(ds, batch_size, workers, subset):
    if ds in ['cifar10', 'cifar100']:
        return DS_TO_FUNC[ds](batch_size, workers, subset)
    
    if subset: raise Exception(f'Subset not supported for the {ds} dataset')
    return DS_TO_FUNC[ds](batch_size, workers)
