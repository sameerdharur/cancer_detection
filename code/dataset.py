from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

class DataSet:
    """
    Note: Ensure data directory is in ImageFolder format.

    """
    
    data_dir = None

    def __init__(self, data_dir):
        self.data_dir = data_dir

    @staticmethod
    def initDataLoaders(data_dir, batch_size):
        data_transforms = DataSet.setUpDataLoaderTransformers()
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val', 'test']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                    shuffle=True, num_workers=4)
                    for x in ['train', 'val']}
        dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size,
                                                    shuffle=False, num_workers=4)
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
        class_names = image_datasets['train'].classes

        return dataloaders, dataset_sizes, class_names

    @staticmethod
    def setUpDataLoaderTransformers(inputSize = 224):
                
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(inputSize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(inputSize),
                transforms.CenterCrop(inputSize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(inputSize),
                transforms.CenterCrop(inputSize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    
        # data_transforms = {
        #     'train': transforms.Compose([
        #         transforms.Resize((inputSize,inputSize)),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #     ]),
        #     'val': transforms.Compose([
        #         transforms.Resize((inputSize,inputSize)),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #     ]),
        #     'test': transforms.Compose([
        #         transforms.Resize((224,224)),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #     ]),
        # }

        return data_transforms