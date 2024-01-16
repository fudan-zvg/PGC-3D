import torch
import torch.nn.functional as F
import cv2
import os
from diffusers import (
    AutoencoderKL,
    StableDiffusionPipeline,
)
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

path = "path to images folder"
imgs_path = os.listdir(path)

imgs_128 = []
imgs_1024 = []
latents = []

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    # "pretrained/SDXL/vae_fp16",
    # local_files_only=True,
    use_safetensors=True,
    torch_dtype=precision_t
)

def encode_imgs(imgs):
    # imgs: [B, 3, H, W]
    imgs = 2 * imgs - 1
    posterior = vae.encode(imgs).latent_dist
    latents = posterior.sample() * vae.config.scaling_factor
    return latents


with torch.no_grad():
    for i in tqdm(range(1000)):
        img = cv2.imread(os.path.join(path, imgs_path[i]))
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB) / 255
        img = torch.tensor(img, dtype=torch.float32, device='cuda')[None, ...]
        img = img.permute(0, 3, 1, 2)
        img_1024 = F.interpolate(img, (1024, 1024), mode='bilinear', align_corners=False)
        img_128 = F.interpolate(img, (128, 128), mode='bilinear', align_corners=False)
        # imgs_1024.append(img_1024)
        imgs_128.append(img_128)

        with torch.cuda.amp.autocast(enabled=True):
            latent = encode_imgs(img_1024)
        latents.append(latent)
        torch.cuda.empty_cache()

data = torch.cat(imgs_128, dim=0).permute(0, 2, 3, 1).reshape(-1, 3)  # [N, 3]
target = torch.cat(latents, dim=0).permute(0, 2, 3, 1).reshape(-1, 4)  # [N, 4]
data_ = torch.cat([data, torch.ones(data.shape[0], 1).cuda()], dim=1)
target_ = torch.cat([target, torch.ones(target.shape[0], 1).cuda()], dim=1)

coef_r2l = torch.linalg.inv(data_.T @ data_ + 0.001 * torch.eye(4).cuda()) @ (data_.T @ target)
coef_l2r = (torch.linalg.inv(target_.T @ target_ + 0.001 * torch.eye(5).cuda())) @ (target_.T @ data)

def normalize(img):
    img = (img - img.min()) / (img.max() - img.min())
    return img


save_path = "2d/vae_linear_approx/sdxl"
os.makedirs(os.path.join(save_path, 'l2r'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'r2l'), exist_ok=True)
for i in range(100):
    x = latents[i][0].permute(1, 2, 0) @ coef_l2r[:4, :] + coef_l2r[4, :]
    rgb_fit = (np.clip((x.detach().cpu().numpy()) * 255, 0, 255)).astype(np.uint8)
    rgb_gt = (imgs_128[i][0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    # plt.imshow(rgb_fit);plt.show()
    # plt.imshow(imgs_128[i][0].permute(1, 2, 0).detach().cpu().numpy());plt.show()

    y = imgs_128[i][0].permute(1, 2, 0) @ coef_r2l[:3, :] + coef_r2l[3, :]
    latent_fit = (normalize(y.detach().cpu().numpy()) * 255).astype(np.uint8)
    latent_gt = (normalize(latents[i][0].permute(1, 2, 0).detach().cpu().numpy()) * 255).astype(np.uint8)
    # plt.imshow(normalize(y.detach().cpu().numpy()));plt.show()
    # plt.imshow(normalize(latents[i][0].permute(1, 2, 0).detach().cpu().numpy()));plt.show()

    cv2.imwrite(os.path.join(save_path, 'l2r', f'{i:04d}_fit.png'),
                cv2.cvtColor(rgb_fit, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(save_path, 'l2r', f'{i:04d}_gt.png'),
                cv2.cvtColor(rgb_gt, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(save_path, 'r2l', f'{i:04d}_fit.png'),
                cv2.cvtColor(latent_fit, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(save_path, 'r2l', f'{i:04d}_gt.png'),
                cv2.cvtColor(latent_gt, cv2.COLOR_RGB2BGR))
