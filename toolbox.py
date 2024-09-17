import argparse
import os
import numpy as np
import ruamel.yaml as yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from skimage.segmentation import mark_boundaries
from skimage.transform import resize
from skimage.segmentation import felzenszwalb
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from art.attacks.poisoning import SleeperAgentAttack
from art.estimators.classification import PyTorchClassifier
from art.metrics import clever_u
from art.metrics.privacy.membership_leakage import SHAPr
from art.utils import to_categorical
from lime import lime_image

from model import Net  # Ensure that the Net class is defined in your model.py
from mydata import load_mydata  # Ensure you have a file named mydata.py containing the load_mydata function
from utils import load_mnist_dataset, load_cifar10_dataset, select_trigger_train, save_poisoned_data, add_trigger_patch, to_one_hot, ToRGBTransform


def calculate_sclever_scores(nb_classes, model, x_test, device_type='gpu'):
    """Calculate untargeted CLEVER scores."""
    device_type = 'gpu' if torch.cuda.is_available() else 'cpu'
    input_shape = x_test.shape[1:]
    clever_calculator = PyTorchClassifier(
        model=model,
        input_shape=input_shape,
        nb_classes=nb_classes,
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.SGD(model.parameters(), lr=0.1),
        device_type=device_type,
    )

    scores = []
    for i in range(50):  # Use the first 50 test images
        image = x_test[i].numpy()  # Keep batch dimension
        clever_untargeted = clever_u(clever_calculator, image, nb_batches=50, batch_size=10, radius=5, norm=1)
        scores.append(clever_untargeted)

    clever_untargeted_avg = np.mean(scores)
    print('-----------FINAL RESULT-----------')
    print(f"Untargeted CLEVER score: {clever_untargeted_avg}")


def calculate_SHAPr(nb_classes, model, x_train, y_train, x_test, y_test, device_type='gpu'):
    """Calculate SHAPr leakage."""
    device_type = 'gpu' if torch.cuda.is_available() else 'cpu'
    input_shape = x_train.shape[1:]
    classifier = PyTorchClassifier(
        model=model,
        input_shape=input_shape,
        nb_classes=nb_classes,
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.SGD(model.parameters(), lr=0.1),
        device_type=device_type,
    )

    SHAPr_leakage = SHAPr(classifier, x_train.numpy(), y_train.numpy(), x_test.numpy(), y_test.numpy(), enable_logging=True)
    print("Average SHAPr leakage: ", np.mean(SHAPr_leakage))


def poison(nb_classes, test, trigger_path, patch_size, x_train, x_test, y_train, y_test, min_, max_, model):
    """Perform data poisoning attack."""
    device_type = 'gpu' if torch.cuda.is_available() else 'cpu'
    x_train_np = x_train.numpy().astype(np.float32)
    x_test_np = x_test.numpy().astype(np.float32)
    y_train_np = y_train.numpy()
    y_test_np = y_test.numpy()

    y_train_np = to_one_hot(y_train_np, nb_classes)
    y_test_np = to_one_hot(y_test_np, nb_classes)

    min_, max_ = float(min_), float(max_)
    mean, std = x_train_np.mean(), x_train_np.std()

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    model_art = PyTorchClassifier(
        model=model,
        input_shape=x_train_np.shape[1:],
        loss=loss_fn,
        optimizer=optimizer,
        nb_classes=nb_classes,
        clip_values=(min_, max_),
        preprocessing=(mean, std),
        device_type=device_type,
    )

    # Handle trigger patch
    img = Image.open(trigger_path)
    input_channel = x_train_np.shape[1:][0]

    patch = resize(np.asarray(img), (patch_size, patch_size, input_channel))  # Ensure the patch has the same dimensions as the data
    patch = np.transpose(patch, (2, 0, 1))

    class_source = 0
    class_target = 1
    K = 1000

    x_trigger, y_trigger, index_target = select_trigger_train(x_train_np, y_train_np.argmax(axis=1), K, class_source, class_target)
    
    attack = SleeperAgentAttack(
        classifier=model_art,
        percent_poison=0.50,
        max_trials=1,
        max_epochs=500,
        earning_rate_schedule=(np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5]), [250, 350, 400, 430, 460]),
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
        device_name=device_type,
    )
    ''' above code for quick check 
    attack = SleeperAgentAttack(
        classifier=model_art,
        percent_poison=0.1,  # Reduce poisoning proportion to 10%
        max_trials=1,        # Keep number of trials to 1
        max_epochs=1,        # Reduce max training epochs to 1
        learning_rate_schedule=(np.array([1e-1]), [0]),  # Use a single learning rate
        epsilon=8/255,       # Slightly reduce perturbation strength
        batch_size=32,       # Reduce batch size
        verbose=1,           # Turn off verbose output
        indices_target=index_target,  # Use only the first 10 target indices
        patching_strategy="fixed",         # Use fixed trigger patch location
        selection_strategy="random",       # Randomly select samples for poisoning
        patch=patch,
        retraining_factor=1,   # No additional retraining
        model_retrain=False,   # Skip model retraining
        model_retraining_epoch=1,
        retrain_batch_size=32,
        class_source=class_source,
        class_target=class_target,
        device_name=device_type
    )
    '''
    # Generate poisoned data
    x_poison, y_poison = attack.poison(x_trigger, y_trigger, x_train_np, y_train_np, x_test_np, y_test_np)
    save_poisoned_data(x_poison, y_poison, class_source, class_target)

    indices_poison = attack.get_poison_indices()
    # np.save('indices_poison.npy', indices_poison)
    # print("indices_poison saved")

    # Check attack effect
    if test:
        # model_art.fit(x_poison, y_poison, batch_size=128, nb_epochs=150, verbose=1)
        model_art.fit(x_poison, y_poison, batch_size=128, nb_epochs=1, verbose=1)
        index_source_test = np.where(y_test_np.argmax(axis=1) == class_source)[0]
        x_test_trigger = x_test_np[index_source_test]
        x_test_trigger = add_trigger_patch(x_test_trigger, trigger_path, patch_size, input_channel, "random")
        result_poisoned_test = model_art.predict(x_test_trigger)

        success_test = (np.argmax(result_poisoned_test, axis=1) == class_target).sum() / result_poisoned_test.shape[0]
        print("Test Success Rate:", success_test)


def explain(nb_classes, num_channels, model):
    """Generate model explanations using LIME."""
    device_type = 'gpu' if torch.cuda.is_available() else 'cpu'

    from skimage.segmentation import felzenszwalb  # Ensure felzenszwalb is imported

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1) if num_channels == 1 else transforms.Lambda(lambda x: x),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if num_channels == 1 else x),
    ])

    dataset = datasets.ImageFolder(root='images_upload', transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    input_shape = (num_channels, dataset[0][0].shape[1], dataset[0][0].shape[2])

    classifier = PyTorchClassifier(
        model=model,
        input_shape=input_shape,
        nb_classes=nb_classes,
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.SGD(model.parameters(), lr=0.1),
        device_type=device_type,
    )

    def predict_fn(images_np):
        # images_np shape: (N, H, W, C)
        # Convert to (N, C, H, W)
        images_tensor = torch.tensor(images_np).permute(0, 3, 1, 2).float()
        # Decide whether to convert the number of channels based on num_channels
        if num_channels == 1 and images_tensor.shape[1] != 1:
            # Convert image to single channel
            images_tensor = images_tensor[:, 0:1, :, :]
        outputs = classifier.predict(images_tensor)
        return outputs

    import os
    output_dir = 'explained_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (images, labels) in enumerate(dataloader):
        images = images.numpy()
        sample = images[0]  # Shape: (C, H, W)
        sample = np.transpose(sample, (1, 2, 0))  # Convert to (H, W, C)
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            sample,
            classifier_fn=predict_fn,
            top_labels=1,
            hide_color=1,
            num_samples=1000,
            # segmentation_fn=felzenszwalb,
            batch_size=1
        )

        temp, mask = explanation.get_image_and_mask(
            label=explanation.top_labels[0],
            positive_only=False,
            num_features=10,
            hide_rest=False,
        )
        img_boundary = mark_boundaries(temp, mask)

        output_filename = f'{output_dir}/explanation_{i}.png'
        plt.imsave(output_filename, img_boundary)
        print(f'Saved image: {output_filename}')


if __name__ == '__main__':
    # Read YAML configuration file
    yaml_loader = yaml.YAML(typ='safe', pure=True)
    with open('config.yaml', 'r') as file:
        config = yaml_loader.load(file)
    load_path = config['model_path']
    trigger_path = config['trigger_path']

    # Define ArgumentParser instance
    parser = argparse.ArgumentParser(
        prog='get_clever_scores',
        description='Calculate CLEVER scores',
    )

    parser.add_argument('-d', '--dataset', required=True, help='Dataset to import (mnist, cifar10, mydata)')
    parser.add_argument('-t', '--task_need', required=True, type=str, help='Task to perform: robustness, privacy, poison, or explain')
    parser.add_argument('-c', '--nb_classes', required=True, type=int, help='Number of classes')
    parser.add_argument('-s', '--patch_size', type=int, help='Patch size for poison data, only for data poisoning task')
    parser.add_argument('-test', '--check_attack_effect', action='store_true', help='Check attack effect, only for data poisoning task')
    parser.add_argument('-ch', '--num_channels', type=int, help='Number of channels in uploaded images, only for model explanation task')

    args = parser.parse_args()

    # Load dataset
    if args.dataset == 'mnist':
        x_train, y_train, x_test, y_test, min_value, max_value = load_mnist_dataset()
    elif args.dataset == 'cifar10':
        print('Loading CIFAR-10 dataset...')
        x_train, y_train, x_test, y_test, min_value, max_value = load_cifar10_dataset()
    elif args.dataset == 'mydata':
        x_train, y_train, x_test, y_test, min_value, max_value = load_mydata()
    else:
        raise ValueError('Please specify a correct dataset: mnist, cifar10, or mydata.')

    # Define input shape
    input_shape = tuple(x_train.shape[1:])

    # Load model
    model = Net()  # Ensure that the Net class is defined in your model.py
    model.load_state_dict(torch.load(load_path))
    model.eval()

    # Call the corresponding function based on the provided task
    if args.task_need == 'robustness':
        calculate_sclever_scores(args.nb_classes, model, x_test)
    elif args.task_need == 'privacy':
        calculate_SHAPr(args.nb_classes, model, x_train, y_train, x_test, y_test)
    elif args.task_need == 'poison':
        if args.patch_size is None:
            raise ValueError('Please provide --patch_size for data poisoning task.')
        poison(args.nb_classes, args.check_attack_effect, trigger_path, args.patch_size, x_train, x_test, y_train, y_test, min_value, max_value, model)
    elif args.task_need == 'explain':
        if args.num_channels is None:
            raise ValueError('Please provide --num_channels for model explanation task.')
        explain(args.nb_classes, args.num_channels, model)
    else:
        print('Please specify a correct task: robustness, privacy, poison, or explain.')
