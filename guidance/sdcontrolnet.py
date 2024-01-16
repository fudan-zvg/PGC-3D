from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler
)
from diffusers.utils.import_utils import is_xformers_available

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from guidance.guidance_utils import SpecifyGradient, noise_norm


class SDControlNet(nn.Module):
    def __init__(self, device, opt):
        super().__init__()
        self.device = device
        self.opt = opt

        print(f'[INFO] loading stable diffusion v1.5 controlnet...')

        precision_t = torch.float16 if opt.fp16 else torch.float32
        variant = "fp16" if opt.fp16 else None

        # multi-controlnets
        controlnets = []
        for control_type in opt.control_type:
            control_kwargs = {
                # "local_files_only": True,
                "use_safetensors": True,
                "torch_dtype": precision_t,
                "variant": variant,
            }
            if 'depth' in control_type:
                model_key = "fusing/stable-diffusion-v1-5-controlnet-depth"
            elif 'normal' in control_type:
                model_key = "fusing/stable-diffusion-v1-5-controlnet-normal"
            elif 'shuffle' in control_type:
                model_key = "lllyasviel/control_v11e_sd15_shuffle"
            else:
                model_key = control_type

            controlnet = ControlNetModel.from_pretrained(
                model_key,
                **control_kwargs
            )
            controlnets.append(controlnet)

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnets,
            torch_dtype=precision_t,
            # local_files_only=True
        )

        if is_xformers_available():
            pipe.enable_xformers_memory_efficient_attention()
        pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.controlnet = pipe.controlnet
        self.scheduler = pipe.scheduler

        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        print(f'[INFO] loaded stable diffusion v1.5 controlnet!')

    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def train_step(self, text_embeddings, pred_rgb, cond_imgs,
                   guidance_scale=100,
                   latents=None,
                   t_range=(0.02, 0.98),
                   t=None,
                   grad_scale=1.0,
                   weight_choice=0):

        B = pred_rgb.size(0)
        if t is None:
            min_step = int(self.num_train_timesteps * t_range[0])
            max_step = int(self.num_train_timesteps * t_range[1])
            t = torch.randint(min_step, max_step + 1, [B], dtype=torch.long, device=self.device)
        else:
            t = torch.tensor(t, dtype=torch.long, device=self.device).expand(B)

        if latents is None:
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_512)
        else:
            latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)

        cond_imgs_ = []
        for i in range(len(cond_imgs)):
            cond_img = cond_imgs[i]
            if cond_img.shape[1] != 3:  # depth
                depth_min = torch.amin(cond_img, dim=[1, 2, 3], keepdim=True)
                depth_max = torch.amax(cond_img, dim=[1, 2, 3], keepdim=True)
                cond_img = (cond_img - depth_min) / (depth_max - depth_min)
                cond_img = F.interpolate(cond_img, (512, 512), mode='bilinear', align_corners=False)
                cond_img = cond_img.expand([B, 3, 512, 512])

            cond_imgs_.append(cond_img.repeat(2, 1, 1, 1))

        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            t_ = torch.cat([t] * 2, dim=0)

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                t_,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=cond_imgs_,
                conditioning_scale=self.opt.control_scale,
                return_dict=False,
            )

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t_,
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t)
        if weight_choice == 0:
            w = (1 - self.alphas[t])[:, None, None, None]  # sigma_t^2
        elif weight_choice == 1:
            w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
            w = w[:, None, None, None]
        elif weight_choice == 2:  # check official code in Fantasia3D
            w = 1 / (1 - self.alphas[t])
            w = w[:, None, None, None]
        else:
            w = 1

        if self.opt.csd:
            grad = w * (noise_pred_text - noise_pred_uncond) * grad_scale
        elif self.opt.ssd:
            h = w * (noise_pred_text - noise_pred_uncond)
            r = (noise_pred_text * noise).sum() / noise_norm(noise)
            noise_tilde = noise_pred_text - r * noise
            E_ssd = noise_norm(h) / noise_norm(noise_tilde) * noise_tilde
            grad = h
            grad[t <= self.opt.ssd_M] = E_ssd[t <= self.opt.ssd_M]
            grad = grad * grad_scale
        else:
            grad = w * (noise_pred - noise) * grad_scale
        grad = torch.nan_to_num(grad)

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        loss = SpecifyGradient.apply(latents, grad)

        return loss

    # TODO: support multi-control
    def train_step_BGT(self, text_embeddings, pred_rgb, cond_img,
                       guidance_scale=100, t=0.98, r=0.25, w_rgb=0.1):

        latents = self.encode_imgs(pred_rgb)
        B = pred_rgb.size(0)

        if cond_img.shape[1] != 3:  # depth
            depth_min = torch.amin(cond_img, dim=[1, 2, 3], keepdim=True)
            depth_max = torch.amax(cond_img, dim=[1, 2, 3], keepdim=True)
            cond_img = (cond_img - depth_min) / (depth_max - depth_min)
            B = cond_img.shape[0]
            cond_img = cond_img.expand([B, 3, 512, 512])

        with torch.no_grad():

            noise = torch.randn_like(latents)
            timestep = torch.tensor([t*self.num_train_timesteps] * B, dtype=torch.long, device=self.device)
            latents_noisy = self.scheduler.add_noise(latents, noise, timestep)

            num_inference_steps = math.ceil(t/r)
            self.scheduler.num_inference_steps = num_inference_steps
            step_ratio = self.num_train_timesteps * r
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            timesteps += int(t*self.num_train_timesteps - (num_inference_steps-1) * step_ratio)
            self.scheduler.timesteps = torch.from_numpy(np.append(timesteps, [0])).to(self.device)
            for i, t in enumerate(self.scheduler.timesteps):
                latent_model_input = torch.cat([latents_noisy] * 2)
                timestep = t.expand(B)
                t_ = torch.cat([timestep] * 2, dim=0)

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input,
                    t_,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=cond_img,
                    return_dict=False,
                )

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t_,
                    encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_noisy = self.scheduler.step(noise_pred, t, latents_noisy)['prev_sample']

            if w_rgb > 0:
                img_gt = self.decode_latents(latents_noisy)

        loss = torch.square(latents - latents_noisy).mean()
        if w_rgb > 0:
            loss += w_rgb * torch.square(pred_rgb - img_gt).mean()

        return loss

    def decode_latents(self, latents):

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--save_name', type=str)
    parser.add_argument('--cond_path', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    device = torch.device('cuda')

    from transformers import pipeline
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
    from PIL import Image
    import numpy as np
    import torch
    from diffusers.utils import load_image
    import cv2
    import os

    image = cv2.imread(opt.cond_path) / 255
    image = torch.tensor(image).unsqueeze(0).permute(0, 3, 1, 2)

    controlnet = ControlNetModel.from_pretrained(
        "fusing/stable-diffusion-v1-5-controlnet-depth",
        local_files_only=True,
        torch_dtype=torch.float32
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        local_files_only=True,
        safety_checker=None,
        torch_dtype=torch.float32
    ).to('cuda')

    #pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    for i in range(1):
        prompt = opt.prompt
        image = pipe(prompt, image, num_inference_steps=50).images[0]
        save_name = opt.save_name
        os.makedirs(os.path.join('2d', save_name), exist_ok=True)
        image.save(f'./2d/{save_name}/{i:04d}.png')
