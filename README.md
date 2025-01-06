# FashionBBDM
## Prerequisites
We tested the code on Ubuntu 22.04. You can use the following command to clone the code and install the required dependencies.
```
git clone https://github.com/wzm206/FashionBBDM.git
cd FashionBBDM
pip install -r requirements.txt
```
## Quick start
We prepared the required conditional images for inference demonstrations in the `demo_input` folder. You need to download the pre trained model that has already been trained from the link below first.

[FashionBBDM](https://huggingface.co/wzm206/FashionBBDM/tree/main)

Create the `model_weight` folder in the root directory of the project and move all files of the pre-trained model into it. Then run `python sample.py`. The generated clothing images are output to the `output` folder by default. 
