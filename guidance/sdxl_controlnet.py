import math
import numpy as np
import cv2
import os

from transformers import logging
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.utils.import_utils import is_xformers_available

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

from guidance.guidance_utils import SpecifyGradient, noise_norm


class SDXLControlNet(nn.Module):
    def __init__(self, device, opt, vram_O=False, hf_key=None):
        super().__init__()

        self.device = device
        self.sd_version = opt.sd_version
        self.opt = opt

        print(f'[INFO] loading SDXL controlnet...')

        with open('./TOKEN', 'r') as f:
            token = f.read().replace('\n', '')  # remove the last \n!
            print(f'[INFO] loaded hugging face access token from ./TOKEN!')

        precision_t = torch.float16 if opt.fp16 else torch.float32
        variant = "fp16" if opt.fp16 else None

        # base xl model key
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == 'xl-1.0':
            model_key = "stabilityai/stable-diffusion-xl-base-1.0"
        else:
            model_key = "stabilityai/stable-diffusion-xl-base-0.9"

        # multi-controlnets
        controlnets = []
        for control_type in opt.control_type:
            if 'depth' in control_type:
                # controlnet_key = "pretrained/SDXL/depth_control"
                controlnet_key = "diffusers/controlnet-depth-sdxl-1.0"
            else:
                controlnet_key = control_type

            controlnet = ControlNetModel.from_pretrained(
                controlnet_key,
                variant=variant,
                use_safetensors=True,
                # local_files_only=True,
                torch_dtype=precision_t,
            )
            controlnets.append(controlnet)

        # Create model
        cache_dir = "pretrained/SDXL"
        general_kwargs = {
            "cache_dir": cache_dir,
            "torch_dtype": precision_t,
            "use_safetensors": True,
            "variant": variant,
            "local_files_only": True,
            "use_auth_token": token,
        }

        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            # "pretrained/SDXL/vae_fp16",
            # local_files_only=True,
            use_safetensors=True,
            torch_dtype=precision_t
        )

        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            model_key,
            controlnet=controlnets,
            vae=vae,
            **general_kwargs
        )

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            if is_xformers_available():
                pipe.enable_xformers_memory_efficient_attention()
            pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.text_encoder = pipe.text_encoder
        self.text_encoder_2 = pipe.text_encoder_2
        self.unet = pipe.unet
        self.controlnet = pipe.controlnet
        # self.scheduler = pipe.scheduler
        if self.opt.scheduler == "ddim":
            Scheduler = DDIMScheduler
        elif self.opt.scheduler == "ddpm":
            Scheduler = DDPMScheduler
        elif self.opt.scheduler == "dpm":
            Scheduler = DPMSolverMultistepScheduler
        else:
            self.scheduler = pipe.scheduler
            Scheduler = None
        if Scheduler is not None:
            self.scheduler = Scheduler.from_pretrained(
                model_key,
                cache_dir=cache_dir,
                subfolder="scheduler",
                use_auth_token=token,
                torch_dtype=precision_t,
            )
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        print(f'[INFO] loaded SDXL controlnet!')

    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        prompt_embeds_list = []
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            with torch.no_grad():
                prompt_embeds = text_encoder(text_inputs.input_ids.to(self.device),
                                             output_hidden_states=True, )
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]
                prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # Do the same for unconditional embeddings
        negative_prompt_embeds_list = []
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            uncond_input = tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            with torch.no_grad():
                negative_prompt_embeds = text_encoder(uncond_input.input_ids.to(self.device),
                                                      output_hidden_states=True, )
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]
                negative_prompt_embeds_list.append(negative_prompt_embeds)

        negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def train_step(self, text_embeddings, pred_rgb, cond_imgs,
                   guidance_scale=100,
                   latents=None,
                   t_range=(0.02, 0.98),
                   t=None,
                   grad_scale=1.0,
                   weight_choice=0):

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = text_embeddings

        B = pred_rgb.size(0)

        add_text_embeds = pooled_prompt_embeds
        res = 1024  # if self.opt.latent else self.opt.res_fine
        add_time_ids = self._get_add_time_ids(
            (res, res), (0, 0), (res, res), dtype=prompt_embeds.dtype
        ).repeat_interleave(B, dim=0)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0).to(self.device)

        if latents is None:
            pred_rgb_1024 = F.interpolate(pred_rgb, (1024, 1024), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_1024)
        else:
            latents = F.interpolate(latents, (128, 128), mode='bilinear', align_corners=False)

        # timestep
        if t is None:
            min_step = int(self.num_train_timesteps * t_range[0])
            max_step = int(self.num_train_timesteps * t_range[1])
            t = torch.randint(min_step, max_step + 1, [B], dtype=torch.long, device=self.device)
        else:
            t = torch.tensor(t, dtype=torch.long, device=self.device).expand(B)

        cond_imgs_ = []
        for i in range(len(cond_imgs)):
            cond_img = cond_imgs[i]
            if cond_img.shape[1] != 3:  # depth
                depth_min = torch.amin(cond_img, dim=[1, 2, 3], keepdim=True)
                depth_max = torch.amax(cond_img, dim=[1, 2, 3], keepdim=True)
                cond_img = (cond_img - depth_min) / (depth_max - depth_min)
                cond_img = F.interpolate(cond_img, (1024, 1024), mode='bilinear', align_corners=False)
                cond_img = cond_img.expand([B, 3, 1024, 1024])

            cond_imgs_.append(cond_img.repeat(2, 1, 1, 1))

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            t_ = torch.cat([t] * 2, dim=0)
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                t_,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=cond_imgs_,
                conditioning_scale=self.opt.control_scale,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )

            noise_pred = self.unet(
                latent_model_input,
                t_,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

        if self.opt.cfg_rescale:
            rescale = 0.5
            std_pos = noise_pred_text.std([1, 2, 3], keepdim=True)
            std_cfg = noise_pred.std([1, 2, 3], keepdim=True)
            factor = std_pos / std_cfg
            factor = rescale * factor + (1 - rescale)
            noise_pred = noise_pred * factor

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

        if self.opt.grad_clip_latent >= 0:
            grad = grad.clamp(-self.opt.grad_clip_latent, self.opt.grad_clip_latent)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(latents, grad)

        return loss

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def decode_latents(self, latents, refiner_vae=False):

        if refiner_vae:
            vae = self.refiner.vae.to(torch.float32)
        else:
            vae = self.vae

        latents = 1 / vae.config.scaling_factor * latents

        with torch.no_grad():
            imgs = vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    import os
    from tqdm import tqdm
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', default='An astronaut riding a green horse', type=str)
    parser.add_argument('--save_name', type=str)
    parser.add_argument('--cond_path', type=str)
    parser.add_argument('--scheduler', default='none', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.1', help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    parser.add_argument('-H', type=int, default=1024)
    parser.add_argument('-W', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    # seed_everything(opt.seed)

    with open('./TOKEN', 'r') as f:
        token = f.read().replace('\n', '')  # remove the last \n!
        print(f'[INFO] loaded hugging face access token from ./TOKEN!')

    device = torch.device('cuda')
    prompt = opt.prompt

    controlnet = ControlNetModel.from_pretrained(
        "pretrained/SDXL/depth_control",
        local_files_only=True,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    ).to("cuda")
    vae = AutoencoderKL.from_pretrained(
        "pretrained/SDXL/vae_fp16",
        use_safetensors=True,
        local_files_only=True,
        torch_dtype=torch.float16
    )
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-0.9',
        cache_dir='pretrained/SDXL',
        local_files_only=True,
        controlnet=controlnet,
        vae=vae,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
        use_auth_token=token
    ).to("cuda")

    pipe.to("cuda")
    if is_xformers_available():
        pipe.enable_xformers_memory_efficient_attention()


    def get_depth_map(image):
        depth_map = torch.nn.functional.interpolate(
            image,
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)

        image = depth_map.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image


    image = cv2.imread(opt.cond_path) / 255
    image = torch.tensor(image).unsqueeze(0).permute(0, 3, 1, 2)
    image = get_depth_map(image)

    for i in range(1):
        prompt = opt.prompt
        image = pipe(prompt,
                     image=image,
                     num_inference_steps=30,
                     controlnet_conditioning_scale=0.75).images[0]
        save_name = opt.save_name
        os.makedirs(os.path.join('2d', save_name), exist_ok=True)
        image.save(f'./2d/{save_name}/{i:04d}.png')
