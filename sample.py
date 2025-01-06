
from diffusers.utils import load_image
import torch
import os
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import transforms
from tqdm.auto import tqdm
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, PNDMScheduler
from scheduler.BBDMScheduler import BBDMScheduler
from model.resnet_down import PoseEncoder

model_path = "model_weight"
# sd_model_path = "logs/v13_test/checkpoint-400"

vae = AutoencoderKL.from_pretrained(
    model_path, subfolder="vae",
)
unet = UNet2DConditionModel.from_pretrained(
    model_path, subfolder="unet", revision=None
)
resnet_down = PoseEncoder()
resnet_down.load_state_dict(torch.load(os.path.join(model_path, "resnet_down.pt")))

# Freeze vae and text_encoder and set unet to trainable
vae.requires_grad_(False)
unet.requires_grad_(False)
resnet_down.requires_grad_(False)
resnet_down.eval()
unet.eval()

to_pil = transforms.functional.to_pil_image
untransform = transforms.Compose([
    transforms.Normalize(mean=[-1, -1, -1], 
                        std=[1/0.5, 1/0.5, 1/0.5]),
])

# Preprocessing the datasets.
image_transforms = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

conditioning_image_transforms = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

def preprocess_train(examples):
    conditioning_images = [image.convert("RGB") for image in examples["conditioning_pixel_values"]]
    conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]

    # examples["pixel_values"] = images
    examples["conditioning_pixel_values"] = torch.tensor(np.array(conditioning_images)).to(device)

    return examples


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
    }

device = "cuda:0"
# Move text_encode and vae to gpu and cast to weight_dtype
scheduler_fashionbbdm = BBDMScheduler()

vae.to(device)
unet.to(device)
resnet_down.to(device)

img_list = os.listdir("demo_input")

for img_name in img_list:
    progress_bar = tqdm(total=200)
    progress_bar.set_description("test_sample")
    control_image = load_image(os.path.join("demo_input", img_name))
    examples = {"conditioning_pixel_values": [control_image]}
    batch = preprocess_train(examples)
    
    
    with torch.no_grad():
        # encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]
        noise_scheduler = PNDMScheduler(num_train_timesteps=1000, 
                                        beta_end=0.012,
                                        beta_schedule="scaled_linear", 
                                        beta_start=0.00085,
                                        steps_offset=1, 
                                        skip_prk_steps=True,
                                        )
        noise_scheduler.set_timesteps(199)
        # Convert images to latent space
        
        latents_condition = vae.encode(batch["conditioning_pixel_values"]).latent_dist.sample()
        latents_condition = latents_condition * vae.config.scaling_factor
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents_condition, device=device)
        input = latents_condition
        i = 0
        for t in noise_scheduler.timesteps:
            
            noisy_latents8 = torch.cat([input, latents_condition], dim=1)
            add_features = resnet_down(noisy_latents8)
            clip_out = torch.zeros((1, 77, 768), device=device)
            noisy_residual = unet(input, t, clip_out,
                                  down_block_additional_residuals=add_features, return_dict=False)[0]
            # 这一步最重要
            # input = noise_scheduler.step(noisy_residual, t, input).prev_sample
            input, _ = scheduler_fashionbbdm.p_sample(latents_condition, noisy_residual, i, input)
            progress_bar.update(1)
            i+=1
            

    result = input
    images_fake = vae.decode(input/vae.config.scaling_factor).sample

    
    img_fake = to_pil(untransform(images_fake[0]).clip(0, 1.0))
    num = img_name.split(".")[0]
    img_fake.save(os.path.join("output", num+".png"))
    # break
