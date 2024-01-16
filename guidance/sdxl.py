import math
import numpy as np
import cv2
import os

from transformers import logging
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
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


class StableDiffusionXL(nn.Module):
    def __init__(self, device, opt, vram_O=False, hf_key=None, use_refiner=False):
        super().__init__()

        self.device = device
        self.sd_version = opt.sd_version
        self.opt = opt

        print(f'[INFO] loading SDXL...')

        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
            model_key_refiner = hf_key
        elif 'turbo' in self.sd_version:
            model_key = "stabilityai/sdxl-turbo"
            use_refiner = False
        elif self.sd_version == 'xl-1.0':
            model_key = "stabilityai/stable-diffusion-xl-base-1.0"
            model_key_refiner = "stabilityai/stable-diffusion-xl-refiner-1.0"
        else:
            model_key = "stabilityai/stable-diffusion-xl-base-0.9"
            model_key_refiner = "stabilityai/stable-diffusion-xl-refiner-0.9"

        with open('./TOKEN', 'r') as f:
            token = f.read().replace('\n', '')  # remove the last \n!
            print(f'[INFO] loaded hugging face access token from ./TOKEN!')

        precision_t = torch.float16 if opt.fp16 else torch.float32
        variant = "fp16" if opt.fp16 else None

        # Create model
        cache_dir = "pretrained/SDXL"
        general_kwargs = {
            "cache_dir": cache_dir,
            "torch_dtype": precision_t,
            "use_safetensors": True,
            "variant": variant,
            # "local_files_only": True,
            "use_auth_token": token,
        }
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            # "pretrained/SDXL/vae_fp16",
            # local_files_only=True,
            use_safetensors=True,
            torch_dtype=precision_t
        )
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_key,
            vae=vae,
            **general_kwargs
        )

        if use_refiner:
            refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_key_refiner,
                                                                       **general_kwargs)
            self.refiner = refiner.to("cuda")

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

        print(f'[INFO] loaded SDXL!')

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

    def train_step(self, text_embeddings, pred_rgb,
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

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            t_ = torch.cat([t] * 2, dim=0)
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            noise_pred = self.unet(
                latent_model_input,
                t_,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
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

        if self.opt.grad_clip_latent >= 0:
            grad = grad.clamp(-self.opt.grad_clip_latent, self.opt.grad_clip_latent)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(latents, grad)

        return loss

    def set_timesteps(self, num_inference_steps, last_t):
        """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, optional):
                the device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        # Clipping the minimum of all lambda(t) for numerical stability.
        # This is critical for cosine (squaredcos_cap_v2) noise schedule.
        clipped_idx = torch.searchsorted(torch.flip(self.scheduler.lambda_t, [0]),
                                         self.scheduler.config.lambda_min_clipped)
        last_timestep = ((last_t - clipped_idx).cpu().numpy()).item()

        # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
        if self.scheduler.config.timestep_spacing == "linspace":
            timesteps = (
                np.linspace(0, last_timestep - 1, num_inference_steps + 1).round()[::-1][:-1].copy().astype(np.int64)
            )
        elif self.scheduler.config.timestep_spacing == "leading":
            step_ratio = last_timestep // (num_inference_steps + 1)
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (np.arange(0, num_inference_steps + 1) * step_ratio).round()[::-1][:-1].copy().astype(np.int64)
            timesteps += self.scheduler.config.steps_offset
        elif self.scheduler.config.timestep_spacing == "trailing":
            step_ratio = self.scheduler.config.num_train_timesteps / num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = np.arange(last_timestep, 0, -step_ratio).round().copy().astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.scheduler.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
            )

        sigmas = np.array(((1 - self.scheduler.alphas_cumprod) / self.scheduler.alphas_cumprod) ** 0.5)
        self.scheduler.sigmas = torch.from_numpy(sigmas).to(self.device)

        # when num_inference_steps == num_train_timesteps, we can end up with
        # duplicates in timesteps.
        _, unique_indices = np.unique(timesteps, return_index=True)
        timesteps = timesteps[np.sort(unique_indices)]

        self.scheduler.timesteps = torch.from_numpy(timesteps).to(self.device)

        self.scheduler.num_inference_steps = len(timesteps)

        self.scheduler.model_outputs = [
                                           None,
                                       ] * self.scheduler.config.solver_order
        self.scheduler.lower_order_nums = 0

    def train_step_BGT(self, text_embeddings, pred_rgb,
                       guidance_scale=50, t=0.98, r=0.25, w_rgb=0.1, loss_scale=1.):
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = text_embeddings

        B = pred_rgb.size(0)

        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            (1024, 1024), (0, 0), (1024, 1024), dtype=prompt_embeds.dtype
        ).repeat_interleave(B, dim=0)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0).to(self.device)

        pred_rgb_1024 = F.interpolate(pred_rgb, (1024, 1024), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(pred_rgb_1024)

        with torch.no_grad():

            noise = torch.randn_like(latents)
            timestep = torch.tensor([t * self.num_train_timesteps] * B, dtype=torch.long, device=self.device)
            latents_noisy = self.scheduler.add_noise(latents, noise, timestep)

            self.set_timesteps(math.ceil(t / r), int(t * self.num_train_timesteps))
            for i, timestep in enumerate(self.scheduler.timesteps):
                # if i == 1:
                #     w_rgb = -1
                #     break

                latent_model_input = torch.cat([latents_noisy] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)

                t_ = torch.cat([timestep.expand(B)] * 2, dim=0)
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                noise_pred = self.unet(
                    latent_model_input,
                    t_,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_noisy = self.scheduler.step(noise_pred, timestep, latents_noisy)['prev_sample']

            if w_rgb > 0:
                img_gt = self.decode_latents(latents_noisy)

                img_npy = img_gt.permute(0, 2, 3, 1)[0].detach().cpu().numpy()
                save_path = os.path.join(self.opt.workspace, 'bgt_test_img')
                os.makedirs(save_path, exist_ok=True)
                cv2.imwrite(os.path.join(save_path, f'{int(t * 1000)}.png'),
                            cv2.cvtColor((img_npy * 255).round(), cv2.COLOR_RGB2BGR))

        loss = torch.square(latents - latents_noisy).sum()
        if w_rgb > 0:
            loss = w_rgb * torch.square(pred_rgb_1024 - img_gt).sum()

        return loss * loss_scale

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
