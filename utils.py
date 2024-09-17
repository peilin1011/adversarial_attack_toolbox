import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

from tqdm import trange
import numpy as np
import os, sys
import pdb
from PIL import Image
from numpy import asarray
from skimage.transform import resize
import random

from art import metrics
from torch import Tensor
import matplotlib.pyplot as plt
import torchvision
from torchvision import models, transforms

#from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from art.utils import load_dataset
from art import metrics
from art.estimators.classification import PyTorchClassifier
from art.metrics.privacy.membership_leakage import SHAPr
from art.attacks.poisoning.sleeper_agent_attack import SleeperAgentAttack
from art.utils import to_categorical

from lime import lime_image
# from skimage.segmentation import mark_boundaries

def load_mnist_dataset():
    # Define dataset transformations (data normalization, etc.)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load CIFAR-10 training and testing sets
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Extract the entire training and test sets
    x_train, y_train = torch.utils.data.default_collate(trainset)
    x_test, y_test = torch.utils.data.default_collate(testset)

    min_value = x_train.min()
    max_value = x_train.max()

    return x_train, y_train, x_test, y_test, min_value, max_value

def load_cifar10_dataset():
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )

    # Load CIFAR-10 training and testing sets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Extract the entire training and test sets
    x_train, y_train = torch.utils.data.default_collate(trainset)
    x_test, y_test = torch.utils.data.default_collate(testset)

    min_value = x_train.min()
    max_value = x_train.max()

    return x_train, y_train, x_test, y_test, min_value, max_value

# For task poisoning
def select_trigger_train(x_train, y_train, K, class_source, class_target):
    x_train_ = np.copy(x_train)
    
    # Directly find the indices that match class_source and class_target in the one-dimensional label array
    index_source = np.where(y_train == class_source)[0][:K]
    index_target = np.where(y_train == class_target)[0]
    
    x_trigger = x_train_[index_source]
    y_trigger = to_categorical([class_target], nb_classes=10)
    y_trigger = np.tile(y_trigger, (len(index_source), 1))
    
    return x_trigger, y_trigger, index_target

def save_poisoned_data(x_poison, y_poison, class_source, class_target):
    # Create file names including target and source class
    file_name_x = f"x_poison_source_{class_source}_target_{class_target}.npy"
    file_name_y = f"y_poison_source_{class_source}_target_{class_target}.npy"

    # Save data locally
    np.save(file_name_x, x_poison)
    np.save(file_name_y, y_poison)

    print(f"Poisoned data saved as {file_name_x} and {file_name_y}")

def add_trigger_patch(x_set, trigger_path, patch_size, input_channel, patch_type="fixed"):
    #print(x_set.shape)
    img = Image.open(trigger_path)
    numpydata = asarray(img)
    #print("shape of numpydata",numpydata.shape)
    patch = resize(numpydata, (patch_size, patch_size, input_channel))
    patch = np.transpose(patch, (2, 0, 1))
    print("shape of patch", patch.shape)
    if patch_type == "fixed":
        x_set[:, :, -patch_size:, -patch_size:] = patch
    else:
        for x in x_set:
            x_cord = random.randrange(0,x.shape[1] - patch.shape[1] + 1)
            y_cord = random.randrange(0,x.shape[2] - patch.shape[2] + 1)
            x[:,x_cord:x_cord+patch_size,y_cord:y_cord+patch_size]=patch

    return x_set

def to_one_hot(y, nb_classes):
    return np.eye(nb_classes)[y] 

if __name__ == "__main__":
    pass