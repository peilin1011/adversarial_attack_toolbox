# **Adversarial Attack Toolbox**

This repository contains a set of utilities for evaluating and performing adversarial attacks on machine learning models. It includes functionality for calculating CLEVER scores, assessing privacy vulnerabilities, generating poisoned data, and explaining model predictions using LIME.

## **Features**

- **[CLEVER Scores Calculation](https://openreview.net/pdf?id=BkUHlMZ0b)**: Assess the robustness of a model by calculating CLEVER scores.  
  A higher CLEVER score indicates better network robustness, as the smallest hostile disturbance may have a larger Lp norm. The value range depends on the radius size, which is 0 - 5 by default, and can be modified in the function `utils/compute_untargeted_clever()`.

- **Privacy Assessment**: Evaluate model privacy using [SHAPr leakage metrics](https://arxiv.org/abs/2112.02230).  
  A higher final SHAPr score for a training sample means it is more vulnerable to privacy attacks. The values range from 0 - 1.

- **Data Poisoning**: Perform data poisoning attacks to evaluate model resilience against adversarial examples.  
  The [Hidden Trigger Backdoor Attack Sleeper Agent](https://arxiv.org/pdf/2106.08970) is used for this. The default `class_source` is 0 (source class), and `class_target` is 1 (target class for misclassification). These values can be modified in `toolbox.py/class_source` and `toolbox.py/class_target`.

- **Model Explanation**: Generate model explanations using [LIME](https://github.com/marcotcr/lime).

## **Requirements**

Ensure you have Python 3.7 installed, along with the following packages:

- `torch 1.13.1`
- `torchvision 0.14.1`
- `lime 0.2.0.1`

## **Configuration**

1. **Create a `config.yaml` file** in the same directory with the following format:

    ```yaml
    model_path: 'path/to/your/model.pth'
    trigger_path: 'path/to/trigger/image.png'
    ```

    - `model_path`: Path to your trained model checkpoint.
    - `trigger_path`: Path to the trigger image used for data poisoning.

2. **Define the `Net` class** in `model.py` to represent your neural network architecture.

3. **Store your `.pth` files** in the `models/` directory (e.g., `models/my_model.pth`).

4. **For LIME explanations**: Place images for processing into the `images_upload/class_0/` directory.

5. **For data poisoning**: Place the trigger image into the `trigger/` directory.

6. **Custom Dataset**:
   - **CSV Label File (`labels.csv`)**: This file should contain two columns - the first column is the image file name, and the second column is the corresponding label. Example:
   
     ```csv
     image1.png,0
     image2.png,1
     image3.png,0
     ```
   - **Image folder**: Ensure that the image folder contains all image files with filenames matching the names in the CSV.

## **Usage**

### **1. Calculate CLEVER Scores**

To calculate CLEVER scores for your model, run the following command:

```bash
$ python your_script.py -d <dataset> -t robustness -c <nb_classes>

- <dataset>: Specify your dataset (e.g., "cifar10", "mnist", "mydata").
- <nb_classes>: Replace with the number of classes in your dataset.
```

### **2. Assess Privacy (SHAPr Leakage)**

To evaluate model privacy using SHAPr leakage, run:

```bash
$ python your_script.py -d <dataset>  -t privacy -c <nb_classes>
```
Replace <nb_classes> with the number of classes in your dataset.

### 3. **Perform Data Poisoning**

To generate poisoned data and evaluate the attack effect, execute:

```bash
$ python your_script.py -d <dataset> -t posion -c <nb_classes> -s <patch_size> -test
```

If just need posioned data, execute:

```bash
$ python your_script.py -d <dataset> -t poison -c <nb_classes> -s <patch_size>
```

Replace <nb_classes> with the number of classes in your dataset and <patch_size> with the patch size for the poison data.

### **4. Explain Model Predictions Using LIME**

To generate explanations for model predictions, use:

```bash
$ python your_script.py -d <dataset> -t explain -c <nb_classes> -ch <num_channels>
```
Replace <nb_classes> with the number of classes and <num_channels> with the number of channels in your uploaded images.

## **Example**
### **1. Calculate CLEVER Scores:**
```bash
$ python toolbox.py -d cifar10 -t robustness -c 10
```
### **2. Assess Privacy:**
```bash
$ python toolbox.py -d cifar10 -t privacy -c 10
```
### **3. Perform Data Poisoning:**
```bash
$ python toolbox.py -d cifar10 -t posion -c 10 -s 8 -test
```
### **4. Explain Model Predictions:**
```bash
$ python toolbox.py -d cifar10 -t explain -c 10 -ch 3
```

## **NOTES**
- For data poisoning, adjust the patch_size, learning rates, and other parameters as needed.
- For interpreting single-channel images using LIME, ensure that the input network accepts 3 channels.