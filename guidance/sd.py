import math
import numpy as np

from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import (
    StableDiffusionPipeline,
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

from guidance.guidance_utils import SpecifyGradient, noise_norm, seed_everything


class StableDiffusion(nn.Module):
    def __init__(self, device, opt, vram_O=False, sd_version='2.1', hf_key=None, t_range=[0.02, 0.98]):
        super().__init__()

        self.opt = opt
        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading stable diffusion...')

        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        precision_t = torch.float16 if opt.fp16 else torch.float32

        # Create model
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key,
            torch_dtype=precision_t,
            #local_files_only=True,
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
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        if self.opt.scheduler == "ddim":
            Scheduler = DDIMScheduler
        elif self.opt.scheduler == "ddpm":
            Scheduler = DDPMScheduler
        else:
            Scheduler = DPMSolverMultistepScheduler
        self.scheduler = Scheduler.from_pretrained(
            model_key,
            subfolder="scheduler",
            torch_dtype=precision_t,
            local_files_only=False
        )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        print(f'[INFO] loaded stable diffusion!')

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

    def train_step(self, text_embeddings, pred_rgb,
                   guidance_scale=100,
                   latents=None,
                   t_range=(0.02, 0.98),
                   t=None,
                   grad_scale=1.0,
                   weight_choice=0):

        if latents is None:
            latents = self.encode_imgs(pred_rgb)
        else:
            latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        B = pred_rgb.size(0)
        if t is None:
            min_step = int(self.num_train_timesteps * t_range[0])
            max_step = int(self.num_train_timesteps * t_range[1])
            t = torch.randint(min_step, max_step + 1, [B], dtype=torch.long, device=self.device)
        else:
            t = torch.tensor(t, dtype=torch.long, device=self.device).expand(B)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            t_ = torch.cat([t] * 2, dim=0)
            noise_pred = self.unet(latent_model_input, t_, encoder_hidden_states=text_embeddings).sample

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

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(latents, grad)

        return loss

    def train_step_BGT(self, text_embeddings, pred_rgb, guidance_scale=100,
                       latents=None, t=0.98, r=0.25, w_rgb=0.1):

        if latents is None:
            latents = self.encode_imgs(pred_rgb)
        else:
            latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)

        B = pred_rgb.size(0)
        with torch.no_grad():

            noise = torch.randn_like(latents)
            timestep = torch.tensor([t*self.num_train_timesteps] * B, dtype=torch.long, device=self.device)
            latents_noisy = self.scheduler.add_noise(latents, noise, timestep)

            num_inference_steps = math.ceil(t/r)
            self.scheduler.num_inference_steps = num_inference_steps
            step_ratio = self.num_train_timesteps * r
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            dt = math.ceil(t * self.num_train_timesteps - (num_inference_steps - 1) * step_ratio)
            if dt > 0:
                timesteps += dt
            self.scheduler.timesteps = torch.from_numpy(np.append(timesteps, [0])).to(self.device)
            for i, t in enumerate(self.scheduler.timesteps):
                latent_model_input = torch.cat([latents_noisy] * 2)
                timestep = t.expand(B)
                t_ = torch.cat([timestep] * 2, dim=0)
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t_, encoder_hidden_states=text_embeddings)['sample']

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

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5,
                        latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8),
                                  device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # Save input tensors for UNet
                # torch.save(latent_model_input, "produce_latents_latent_model_input.pt")
                # torch.save(t, "produce_latents_t.pt")
                # torch.save(text_embeddings, "produce_latents_text_embeddings.pt")
                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50,
                      guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts)  # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents,
                                       num_inference_steps=num_inference_steps,
                                       guidance_scale=guidance_scale)  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    import os
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', default='Starbucks 3d logo', type=str)
    parser.add_argument('--negative', default='black', type=str)
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'],
                        help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion(device, opt)

    for index in tqdm(range(10)):
        seed_everything(opt.seed + index)

        imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

        # visualize image
        os.makedirs(os.path.join('2d', str(opt.prompt)), exist_ok=True)
        plt.imsave(os.path.join('2d', str(opt.prompt), str(index) + '.png'), imgs[0])



