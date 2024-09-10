import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model import Net

import ruamel.yaml as yaml
import argparse

from art import metrics
from torch import Tensor
import matplotlib.pyplot as plt
import torchvision
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier

from lime import lime_image
from skimage.segmentation import mark_boundaries
from utils import *
from mydata import *

def calculate_sclever_scores(nb_classes):   
    # Create CLEVER score calculator instance (assuming CleverScoreCalculator is already defined)
    clever_calculator = CleverScoreCalculator(
        model=model,
        input_shape=input_shape,
        nb_classes=nb_classes,
        device_type='cpu'
    )

    # Calculate CLEVER scores
    scores = []
    for i in range(10):  # Use the first 10 test images as an example
        image = x_test[i].numpy()
        clever_untargeted = clever_calculator.compute_untargeted_clever(image)
        scores.append(clever_untargeted)
    
    # Calculate the average score
    clever_untargeted = sum(scores) / len(scores)
        
    print('-----------FINAL RESULT-----------')
    print(f"Untargeted CLEVER score: {clever_untargeted}")

def calculate_SHAPr(nb_classes):
    clever_calculator = CleverScoreCalculator(
        model=model,
        input_shape=input_shape,
        nb_classes=nb_classes,
        device_type='cpu'
    )

    SHAPr_leakage = clever_calculator.computer_SHAPr(x_train, y_train, x_test, y_test)
    print("Average SHAPr leakage: ", np.average(SHAPr_leakage))

def posion(nb_classes, test, trigger_path, patch_size, x_train, x_test, y_train, y_test, min_, max_):
    # Convert dataset to NumPy arrays
    x_train = x_train.numpy().astype(np.float32)
    x_test = x_test.numpy().astype(np.float32)

    y_train = to_one_hot(y_train, nb_classes)  # Convert to one-hot encoding
    y_test = to_one_hot(y_test, nb_classes)

    min_, max_ = min_.numpy(), max_.numpy()  # Convert Tensor to NumPy

    # Directly calculate mean and std without adjusting format
    mean, std = np.mean(x_train), np.std(x_train)

    # Create optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    loss_fn = nn.CrossEntropyLoss()
    model_art = PyTorchClassifier(model,
                                  input_shape=x_train.shape[1:], 
                                  loss=loss_fn, 
                                  optimizer=optimizer, 
                                  nb_classes=nb_classes, 
                                  clip_values=(min_, max_), 
                                  preprocessing=(mean, std)
                                  )

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = Image.open(trigger_path)
    patch = resize(asarray(img), (patch_size, patch_size, 3))
    patch = np.transpose(patch, (2, 0, 1))
    # Change patch to single channel
    patch = np.mean(patch, axis=0, keepdims=True)

    class_source = 0
    class_target = 1
    K = 1000

    x_trigger, y_trigger, index_target = select_trigger_train(x_train, y_train, K, class_source, class_target)
    
    attack = SleeperAgentAttack(
        model_art,
        percent_poison=0.50,
        max_trials=1,
        max_epochs=1,  # Set max_epochs to 1 here
        learning_rate_schedule=(np.array([1e-1, 1e-2]), [25, 40]),  # Modify learning rate schedule to match max_epochs
        epsilon=16/255,
        batch_size=500, 
        verbose=1,
        indices_target=index_target,
        patching_strategy="random",
        selection_strategy="max-norm",
        patch=patch,
        retraining_factor=4,
        model_retrain=True,
        model_retraining_epoch=80,
        retrain_batch_size=128,
        class_source=class_source,
        class_target=class_target,
        device_name=str(device)
    )

    # Generate poison data
    x_poison, y_poison = attack.poison(x_trigger, y_trigger, x_train, y_train, x_test, y_test)

    save_poisoned_data(x_poison, y_poison, class_source, class_target)

    # Compute attack effectiveness
    if test:
        clever_calculator_poisoned = CleverScoreCalculator(
            model=model,
            input_shape=x_train.shape[1:],
            nb_classes=nb_classes,
            optimizer=optimizer,
            device_type='cpu'
        )

        clever_calculator_poisoned.fit(x_poison, y_poison, 
                                       batch_size=128, 
                                       nb_epochs=1,  # Set to 1 for testing
                                       verbose=1)

        index_source_test = np.where(y_test.argmax(axis=1) == class_source)[0]
        x_test_trigger = x_test[index_source_test]
        x_test_trigger = add_trigger_patch(x_test_trigger, trigger_path, patch_size, "random")
        result_poisoned_test = clever_calculator_poisoned.predict(x_test_trigger)

        success_test = (np.argmax(result_poisoned_test, axis=1) ==
                        1).sum() / result_poisoned_test.shape[0]
        print("Test Success Rate:", success_test)

def explain(nb_classes, num_channels):
    # Define transformation
    ######## Just for testing ########
    transform = transforms.Compose(
        [transforms.ToTensor(),
         torchvision.transforms.Grayscale(num_output_channels=1),  # Convert to grayscale image
         ToRGBTransform()
         ])
    ################################
    
    # Load image folder
    dataset = datasets.ImageFolder(root='images_upload', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create an instance of ‘CleverScoreCalculator’
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)

    # Generate explanations using LIME
    for i, (images, labels) in enumerate(dataloader):
        print(images.shape)
        input_shape = images.shape
        clever_calculator = CleverScoreCalculator(
            model=model,
            input_shape=input_shape,
            nb_classes=nb_classes,
            optimizer=optimizer,
            device_type='cpu'
        )

        images = np.squeeze(images.numpy(), axis=0)
        sample = np.transpose(images, (1, 2, 0))  # Convert to (height, width, channels) format
        explanation = clever_calculator.explain_with_lime(sample)

        # Display explanation results
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
        img_boundry2 = mark_boundaries(temp, mask)

        # Find the min and max values of the data
        min_val = np.min(img_boundry2)
        max_val = np.max(img_boundry2)

        # Normalize the data to the 0-1 range
        img_boundry2_normalized = (img_boundry2 - min_val) / (max_val - min_val)

        # Save the image to a file
        output_filename = f'explained_images/explanation_{i}.png'  # File name for the saved image
        plt.imsave(output_filename, img_boundry2_normalized)
        print(f'Saved image: {output_filename}')
        
if __name__ == '__main__':
    # Read the YAML configuration file
    yaml_loader = yaml.YAML(typ='safe', pure=True)
    with open('config.yaml', 'r') as file:
        config = yaml_loader.load(file)
    load_path = config['model_path']
    trigger_path = config['trigger_path']
    
    # Define an ArgumentParser instance
    parser = argparse.ArgumentParser(
        prog='get_clever_scores',
        description='Calculate CLEVER scores',
    )
    
    parser.add_argument('-d', '--dataset', required=True, help='Dataset to import (mnist, cifar10, mydata)')
    parser.add_argument('-t', '--task_need', required=True, type=str, help='Task to perform: robustness, privacy, or poison')
    parser.add_argument('-c', '--nb_classes', required=True, type=int, help='Number of classes')
    parser.add_argument('-s', '--patch_size', type=int, help='Patch size for poison data, just for task Data Poisoning')
    parser.add_argument('-test', '--check_attack_effect', action='store_true', help='Check attack effect, just for task Data Poisoning')
    parser.add_argument('-ch', '--num_channels', type=int, help='Number of channels in uploaded images, just for task Model Explanation') 
    
    args = parser.parse_args()

    # Load dataset
    if args.dataset == 'mnist':
        x_train, y_train, x_test, y_test, min_value, max_value = load_mnist_dataset()
    elif args.dataset == 'cifar10':
        x_train, y_train, x_test, y_test, min_value, max_value = load_cifar10_dataset()
    elif args.dataset == 'mydata':
        x_train, y_train, x_test, y_test, min_value, max_value = load_mydata()
    # Define input_shape
    input_shape = tuple(x_train[0].shape)
    
    # Load model
    model = Net()  # You need to define the Net class
    # model.load_state_dict(torch.load(load_path))
    model.eval()
    
    # Based on the task provided, call the appropriate function
    if args.task_need == 'robustness':
        calculate_sclever_scores(args.nb_classes)
    elif args.task_need == 'privacy':
        calculate_SHAPr(args.nb_classes)
    elif args.task_need == 'poison':
        posion(args.nb_classes, args.check_attack_effect, trigger_path, args.patch_size, x_train, x_test, y_train, y_test, min_value, max_value)
    elif args.task_need == 'explain':
        explain(args.nb_classes, args.num_channels)
