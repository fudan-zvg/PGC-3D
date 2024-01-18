import os
import math
from tqdm import tqdm
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import gc
import time
import io
import cv2

import matplotlib.pyplot as plt
from PIL import Image
from guidance.guidance_utils import SpecifyGradient


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, -1)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, default="old man")
parser.add_argument('--save_path', type=str, default="2d/2d_sds/clip0.1")
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--cfg', type=float, default=100.)
parser.add_argument('--grad_clip_rgb', type=float, default=0.1)
parser.add_argument('--grad_suppress_type', type=int, default=0)
parser.add_argument('--random_init', action='store_true')
parser.add_argument('--img_init_path', type=str, default="data/background.png")
parser.add_argument('--mode', type=str, default="rgb")
parser.add_argument('--res', type=int, default=1024)
opt = parser.parse_args()
opt.scheduler = 'dpm'
opt.grad_clip_latent = -1
opt.fp16 = True

prompt = opt.prompt

if opt.random_init:
    iters = 1000
else:
    iters = 500

# stable diffusion
config = {
    'max_iters': iters,
    'seed': 42,
    'scheduler': 'cosine',
    'mode': opt.mode,
    'prompt_processor_type': 'stable-diffusion-prompt-processor',
    'prompt_processor': {
        'prompt': prompt,
    },
    'guidance_type': 'stable-diffusion-guidance',
    'guidance': {
        'half_precision_weights': False,
        'guidance_scale': opt.cfg,
        'pretrained_model_name_or_path': 'runwayml/stable-diffusion-v1-5',
        'grad_clip': None,
        'view_dependent_prompting': False,
    },
    'image': {
        'width': opt.res,
        'height': opt.res,
    }
}

save_path = opt.save_path
os.makedirs(os.path.join(save_path, 'rgb'), exist_ok=True)
# os.makedirs(os.path.join(save_path, 'grad'), exist_ok=True)
# os.makedirs(os.path.join(save_path, 'grad_norm'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'grad_n'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'grad_norm_n'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'grad_o'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'grad_norm_o'), exist_ok=True)

seed_everything(config['seed'])

# %%

# just need to rerun the cell when you change guidance or prompt_processor
gc.collect()
with torch.no_grad():
    torch.cuda.empty_cache()

from guidance.sdxl import StableDiffusionXL
from guidance.sd import StableDiffusion

guidance = StableDiffusionXL("cuda", opt)
#guidance = StableDiffusion("cuda", opt)

ploss = []


# %%

def figure2image(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def run(config):
    # clear gpu memory
    rgb = None
    grad = None
    vis_grad = None
    vis_grad_norm = None
    loss = None
    optimizer = None
    target = None

    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()

    mode = config['mode']

    w, h = config['image']['width'], config['image']['height']

    if opt.random_init:
        if mode == 'rgb':
            target = nn.Parameter(torch.rand(1, h, w, 3, device=guidance.device))
        else:
            target = nn.Parameter(torch.randn(1, h // 8, w // 8, 4, device=guidance.device))
    else:
        file = opt.img_init_path
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB) / 255
        img = torch.tensor(img, dtype=torch.float32, device=guidance.device)[None, ...]
        if mode == 'rgb':
            target = nn.Parameter(img)
        else:
            with torch.cuda.amp.autocast(enabled=opt.fp16):
                img = guidance.encode_imgs(img.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            target = nn.Parameter(img)

    optimizer = torch.optim.AdamW([target], lr=opt.lr, weight_decay=0, betas=(0.9, 0.999))
    # optimizer = torch.optim.Adam([target], lr=opt.lr, weight_decay=0)
    num_steps = config['max_iters']
    scheduler = get_cosine_schedule_with_warmup(optimizer, 100, int(num_steps * 3)) if config[
                                                                                           'scheduler'] == 'cosine' else None
    scaler = torch.cuda.amp.GradScaler(init_scale=2. ** 10, enabled=opt.fp16)

    rgb = None

    img_array = []
    grad_array = []
    grad_norm_array = []

    text_embeddings = guidance.get_text_embeds([prompt], [''])

    try:
        for step in tqdm(range(num_steps + 1)):
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=opt.fp16):
                if mode == "rgb":
                    latents = None
                else:
                    latents = target.permute(0, 3, 1, 2)

                if opt.random_init:
                    t_max = 0.98
                else:
                    t_max = 0.5
                loss = guidance.train_step(text_embeddings,
                                           target.permute(0, 3, 1, 2),
                                           t_range=[0.02, t_max],
                                           latents=latents)

                # A = torch.tensor([[1.7815, 0.7696, -3.6330, -0.6444],
                #                   [0.7176, 3.8595, 2.8650, 1.7375],
                #                   [-0.1892, -1.8685, 0.1521, -2.8548]]).to("cuda")
                # grad_ = grad.permute(0, 2, 3, 1) @ A.T
                # grad_ = grad_ = F.interpolate(
                #             grad_.permute(0, 3, 1, 2), (opt.res, opt.res), mode="bilinear", align_corners=False
                #         ).permute(0, 2, 3, 1)
                # loss = SpecifyGradient.apply(target, grad_)

            if opt.grad_clip_rgb >= 0:
                def _hook(grad):
                    if opt.fp16:
                        # correctly handle the scale
                        grad_scale = scaler._get_scale_async()
                        clip_value = opt.grad_clip_rgb * grad_scale
                    else:
                        grad_scale = 1.0
                        clip_value = opt.grad_clip_rgb

                    if opt.grad_suppress_type == 0:  # pwclip
                        ratio = 1. / grad.abs() * clip_value
                        ratio[ratio > 1.0] = 1.0
                        grad_ = grad * torch.amin(ratio, dim=[-1], keepdim=True)
                    elif opt.grad_suppress_type == 1:  # clip
                        grad_ = grad.clamp(-clip_value, clip_value)
                    elif opt.grad_suppress_type == 2:  # global scale
                        grad_ = grad / grad.abs().max() * clip_value
                    elif opt.grad_suppress_type == 3:  # sigmoid
                        grad_ = (torch.sigmoid(grad) - 0.5) * clip_value
                    elif opt.grad_suppress_type == 4:  # norm
                        grad_norm = torch.amax(grad.abs(), dim=[-1], keepdim=True)
                        grad_ = clip_value * (grad / (grad_norm + clip_value))
                    else:
                        grad_ = grad
                    return grad_

                target.register_hook(_hook)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            # optimizer_state = scaler._per_optimizer_states[id(optimizer)]
            # if sum(v.item() for v in optimizer_state["found_inf_per_device"].values()):
            #     print("skip")
            scaler.update()

            grad = target.grad
            if scheduler is not None:
                scheduler.step()

            if step % 10 == 0:
                if mode == 'rgb':
                    rgb = target
                    vis_grad = grad[..., :3]
                    vis_grad_norm = grad.norm(dim=-1)
                else:
                    with torch.cuda.amp.autocast(enabled=opt.fp16):
                        rgb = guidance.decode_latents(target.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                    vis_grad = grad
                    vis_grad_norm = grad.norm(dim=-1)

                grad_ = vis_grad_norm
                vis_grad_norm_n = (grad_ - grad_.min()) / (grad_.max() - grad_.min())
                # vis_grad_norm_n = vis_grad_norm / vis_grad_norm.max()
                grad_ = vis_grad
                vis_grad_n = (grad_ - grad_.min()) / (grad_.max() - grad_.min())
                # vis_grad_n = vis_grad / vis_grad.max()
                img_rgb = rgb.clamp(0, 1).detach().squeeze(0).cpu().numpy()
                img_grad_o = (vis_grad+0.5).clamp(0, 1).detach().squeeze(0).cpu().numpy()
                img_grad_norm_o = vis_grad_norm.clamp(0, 1).detach().squeeze(0).cpu().numpy()
                img_grad_n = vis_grad_n.clamp(0, 1).detach().squeeze(0).cpu().numpy()
                img_grad_norm_n = vis_grad_norm_n.clamp(0, 1).detach().squeeze(0).cpu().numpy()

                img = (img_rgb * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(opt.save_path, 'rgb', f'{step:04d}.png'),
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                img = (img_grad_n * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(opt.save_path, 'grad_n', f'{step:04d}.png'),
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                img = (img_grad_norm_n * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(opt.save_path, 'grad_norm_n', f'{step:04d}.png'),
                            img)

                img = (img_grad_o * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(opt.save_path, 'grad_o', f'{step:04d}.png'),
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                img = (img_grad_norm_o * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(opt.save_path, 'grad_norm_o', f'{step:04d}.png'),
                            img)

    except KeyboardInterrupt:
        pass
    finally:
        print("Optimizing Done")

run(config)
