# Controllable Fashion Rendering via Brownian Bridge Diffusion Model with Latent Sketch Encoding
This repository is the official implementation of "Controllable Fashion Rendering via Brownian Bridge Diffusion Model with Latent Sketch Encoding". The work is currently being submitted to [The Visual Computer](https://link.springer.com/journal/371)

![main](./assets/main.png "main")
## Prerequisites
We tested the code on Ubuntu 22.04. You can use the following command to clone the code and install the required dependencies. Our code is based on the [diffuser](https://github.com/huggingface/diffusers) library.
```
git clone https://github.com/wzm206/FashionBBDM.git
cd FashionBBDM
pip install -r requirements.txt
```
## Quick start
We prepared the required conditional images for inference demonstrations in the `demo_input` folder. You need to download the pre trained model that has already been trained from the link below first.

[FashionBBDM](https://huggingface.co/wzm206/FashionBBDM/tree/main)

Create the `model_weight` folder in the root directory of the project and move all files of the pre-trained model into it. Then run `python sample.py`. The generated clothing images are output to the `output` folder by default. The output result should be as shown in the following figure.

![demo](./assets/demo.png "demo")


## Training
### Prepare the dataset
We used the fashion dataset in [FashionGAN: Display your fashion design using Conditional Generative Adversarial Nets](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.13552) and shoe dataset collected in [image-to-image translation with conditional adversarial networks](https://phillipi.github.io/pix2pix/).

Organize the files in the dataset into the following format

```
├── source
│   ├── 1.png
│   └── 2.png
├── target
      ├── 1.png
      └── 2.png
```

If you are using a clothing dataset, you can use the following script:
```
import os
import cv2
img_ori_path = "path"
output_pathA = "path"

img_name_list = os.listdir(img_ori_path)

for i, now_name in enumerate(img_name_list):
    ori_path = os.path.join(img_ori_path, now_name)
    AB = cv2.imread(ori_path)
    A = AB[:, :256]
    B = AB[:, 256:]
    A[96:96+64, 96:96+64] = B[96:96+64, 96:96+64]
    cv2.imwrite(os.path.join(output_pathA, "source", now_name), A)
    cv2.imwrite(os.path.join(output_pathA, "target", now_name), B)
```
### Train
Start training using the following command
```
export MODEL_NAME="/path/to/stable-diffusion-v1-5"
CUDA_VISIBLE_DEVICES=0 accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir /path/to/data \
  --resolution=256 \
  --random_flip \
  --train_batch_size=32 \
  --gradient_checkpointing \
  --learning_rate=2e-5 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="logs/FashionBBDM" \
  --num_train_epochs 200 \
  --checkpointing_steps 1000
```

## Evaluate
You can use the following script to calculate lpips and ssim:
```
import os
import torch
import lpips
from PIL import Image
from torchvision.transforms import ToTensor
from skimage.metrics import structural_similarity as ssim
import numpy as np
from pytorch_fid import fid_score

import torchvision
import torchvision.transforms as transforms

loss_fn = lpips.LPIPS(net='vgg')

def calculate_lpips(img_path1, img_path2):
    img1 = Image.open(img_path1)
    img2 = Image.open(img_path2)

    img1_tensor = ToTensor()(img1).unsqueeze(0)
    img2_tensor = ToTensor()(img2).unsqueeze(0)

    lpips_value = loss_fn(img1_tensor, img2_tensor)
    return lpips_value.item()


folder_path1 = 'compare/'
folder_path2 = 'compare/'

image_paths1 = [os.path.join(folder_path1, img) for img in os.listdir(folder_path1) if img.endswith('.png')]
image_paths2 = [os.path.join(folder_path2, img) for img in os.listdir(folder_path2) if img.endswith('.png')]

assert set(os.path.basename(p) for p in image_paths1) == set(os.path.basename(p) for p in image_paths2), "The two folders must contain images with the same names."

lpips_values = []
ssim_values = []

for img_name in os.listdir(folder_path1):
    if img_name.endswith('.png'):
        img_path1 = os.path.join(folder_path1, img_name)
        img_path2 = os.path.join(folder_path2, img_name)
        lpips_value = calculate_lpips(img_path1, img_path2)
        lpips_values.append(lpips_value)
        print(f'LPIPS between {img_path1} and {img_path2}: {lpips_value}')
        
        img1 = np.array(Image.open(img_path1))
        img2 = np.array(Image.open(img_path2))
        ssim_values.append(ssim(img1, img2, multichannel=True))

average_lpips = sum(lpips_values) / len(lpips_values)
average_ssim = sum(ssim_values) / len(ssim_values)
print(f'Average LPIPS---: {average_lpips}')
print(f'Average SSIM+++: {average_ssim}')
```

Then use the following script to calculate FID

```
inception_model = torchvision.models.inception_v3(pretrained=True)
transform = transforms.Compose([
    # transforms.Resize(299),
    # transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

fid_value = fid_score.calculate_fid_given_paths([folder_path2, folder_path1],device="cuda:0",
                                                batch_size=5, dims=2048 )
print('FID value:', fid_value)
```


## Acknowledgments
This implementation is heavily inspired by [BBDM](https://github.com/xuekt98/BBDM).
