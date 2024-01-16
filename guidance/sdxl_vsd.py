import math
import numpy as np
import cv2
import os
import random
from contextlib import contextmanager

from transformers import logging
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DDPMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    AutoencoderKL
)
from diffusers.utils.import_utils import is_xformers_available
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.utils.import_utils import is_xformers_available

import torch
import torch.nn as nn
import torch.nn.functional as F

# suppress partial model loading warning
logging.set_verbosity_error()


class ToWeightsDType(nn.Module):
    def __init__(self, module: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x):
        return self.module(x).to(self.dtype)


class StableDiffusionXLvsd(nn.Module):
    def __init__(self, device, opt, hf_key=None, use_refiner=True):
        super().__init__()

        self.device = device
        self.sd_version = opt.sd_version
        self.opt = opt

        print(f'[INFO] loading SDXL...')

        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
            model_key_refiner = hf_key
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
            refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                model_key_refiner,
                **general_kwargs
            )
            self.refiner = refiner.to("cuda")

        # lora
        pipe_lora_kwargs = {
            "tokenizer": None,
            "tokenizer_2": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
        }
        pipe_lora_kwargs.update(general_kwargs)
        self.pipe_lora = StableDiffusionXLPipeline.from_pretrained(
            model_key,
            **pipe_lora_kwargs
        )
        del self.pipe_lora.vae
        del self.pipe_lora.text_encoder
        for p in self.pipe_lora.unet.parameters():
            p.requires_grad_(False)

        if is_xformers_available():
            pipe.enable_xformers_memory_efficient_attention()
            self.pipe_lora.enable_xformers_memory_efficient_attention()
        pipe.to(device)
        self.pipe_lora.to(device)

        self.camera_embedding = ToWeightsDType(
            TimestepEmbedding(16, 1280), precision_t
        ).to(device)
        self.pipe_lora.unet.class_embedding = self.camera_embedding

        # set up LoRA layers
        lora_attn_procs = {}
        for name in self.pipe_lora.unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.pipe_lora.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.pipe_lora.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.pipe_lora.unet.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.pipe_lora.unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            ).to(device)

        self.pipe_lora.unet.set_attn_processor(lora_attn_procs)

        self.lora_layers = AttnProcsLayers(self.pipe_lora.unet.attn_processors)
        self.lora_layers._load_state_dict_pre_hooks.clear()
        self.lora_layers._state_dict_hooks.clear()

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.text_encoder = pipe.text_encoder
        self.text_encoder_2 = pipe.text_encoder_2
        self.unet = pipe.unet
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        # self.scheduler = pipe.scheduler
        # self.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        if self.opt.scheduler == "ddim":
            Scheduler = DDIMScheduler
        elif self.opt.scheduler == "ddpm":
            Scheduler = DDPMScheduler
        else:
            Scheduler = DPMSolverMultistepScheduler
        self.scheduler_lora = Scheduler.from_pretrained(
            model_key,
            cache_dir=cache_dir,
            subfolder="scheduler",
            use_auth_token=token,
            torch_dtype=precision_t,
        )
        self.scheduler = Scheduler.from_pretrained(
            model_key,
            cache_dir=cache_dir,
            subfolder="scheduler",
            use_auth_token=token,
            torch_dtype=precision_t,
        )
        self.pipe_lora.scheduler = self.scheduler_lora

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        self.text_embeddings_invd = self.get_text_embeds([self.opt.text], [''])

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

    def get_latents(self, rgb_BCHW, rgb_as_latents=False):
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (128, 128), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (1024, 1024), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.encode_imgs(rgb_BCHW_512)
        return latents

    def forward_unet(self, unet, latents, t, encoder_hidden_states, added_cond_kwargs,
                     class_labels=None, cross_attention_kwargs=None):
        input_dtype = latents.dtype
        return unet(
            latents,
            t,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample.to(input_dtype)

    @contextmanager
    def disable_unet_class_embedding(self, unet):
        class_embedding = unet.class_embedding
        try:
            unet.class_embedding = None
            yield unet
        finally:
            unet.class_embedding = class_embedding

    def train_lora(self, latents, text_embeddings, camera_condition):
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = text_embeddings

        B = latents.shape[0]

        add_text_embeds = pooled_prompt_embeds
        res = 1024 if self.opt.latent else self.opt.res_fine
        add_time_ids = self._get_add_time_ids(
            (res, res), (0, 0), (res, res), dtype=prompt_embeds.dtype
        ).repeat_interleave(B, dim=0).to(self.device)
        prompt_embeds = prompt_embeds.repeat_interleave(B, dim=0)
        add_text_embeds = add_text_embeds.repeat_interleave(B, dim=0)

        latents = latents.detach().repeat(1, 1, 1, 1)

        t = torch.randint(
            int(self.num_train_timesteps * 0.0),
            int(self.num_train_timesteps * 1.0),
            [B],
            dtype=torch.long,
            device=self.device,
        )

        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler_lora.add_noise(latents, noise, t)
        # lora prediction_type == "epsilon"
        target = noise

        # use view-independent text embeddings in LoRA
        if random.random() < 0.1:
            camera_condition = torch.zeros_like(camera_condition)
        noise_pred = self.forward_unet(
            self.pipe_lora.unet,
            noisy_latents,
            t,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={"text_embeds": add_text_embeds, "time_ids": add_time_ids},
            class_labels=camera_condition.view(B, -1),
            cross_attention_kwargs={"scale": 1.0},
        )
        return F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

    def compute_grad_vsd(self, latents, text_embeddings, camera_condition,
                         t_range=(0.02, 0.98),
                         guidance_scale=10,
                         guidance_scale_lora=1.0):
        B = latents.shape[0]

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = text_embeddings
        add_text_embeds = pooled_prompt_embeds
        res = 1024 if self.opt.latent else self.opt.res_fine
        add_time_ids = self._get_add_time_ids(
            (res, res), (0, 0), (res, res), dtype=prompt_embeds.dtype
        ).repeat_interleave(B, dim=0)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0).to(self.device)

        (
            prompt_embeds_invd,
            negative_prompt_embeds_invd,
            pooled_prompt_embeds_invd,
            negative_pooled_prompt_embeds_invd,
        ) = self.text_embeddings_invd
        add_text_embeds_invd = pooled_prompt_embeds_invd
        add_time_ids_invd = self._get_add_time_ids(
            (res, res), (0, 0), (res, res), dtype=prompt_embeds_invd.dtype
        ).repeat_interleave(B, dim=0)
        prompt_embeds_invd = torch.cat([negative_prompt_embeds_invd, prompt_embeds_invd],
                                       dim=0).repeat_interleave(B, dim=0)
        add_text_embeds_invd = torch.cat([negative_pooled_prompt_embeds_invd, add_text_embeds_invd],
                                         dim=0).repeat_interleave(B, dim=0)
        add_time_ids_invd = torch.cat([add_time_ids_invd, add_time_ids_invd], dim=0).to(self.device)

        with torch.no_grad():
            # random timestamp
            min_step = int(self.num_train_timesteps * t_range[0])
            max_step = int(self.num_train_timesteps * t_range[1])
            t = torch.randint(
                min_step,
                max_step + 1,
                [B],
                dtype=torch.long,
                device=self.device,
            )
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            with self.disable_unet_class_embedding(self.unet) as unet:
                cross_attention_kwargs = None
                noise_pred_pretrain = self.forward_unet(
                    unet,
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={"text_embeds": add_text_embeds, "time_ids": add_time_ids},
                    cross_attention_kwargs=cross_attention_kwargs,
                )

            # use view-independent text embeddings in LoRA
            noise_pred_est = self.forward_unet(
                self.pipe_lora.unet,
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=prompt_embeds_invd,
                added_cond_kwargs={"text_embeds": add_text_embeds_invd, "time_ids": add_time_ids_invd},
                class_labels=torch.cat(
                    [
                        camera_condition.view(B, -1),
                        torch.zeros_like(camera_condition.view(B, -1)),
                    ],
                    dim=0,
                ),
                cross_attention_kwargs={"scale": 1.0},
            )

        (
            noise_pred_pretrain_uncond,
            noise_pred_pretrain_text,
        ) = noise_pred_pretrain.chunk(2)

        # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
        noise_pred_pretrain = noise_pred_pretrain_uncond + guidance_scale * (
                noise_pred_pretrain_text - noise_pred_pretrain_uncond
        )

        (
            noise_pred_est_uncond,
            noise_pred_est_text,
        ) = noise_pred_est.chunk(2)

        # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
        noise_pred_est = noise_pred_est_uncond + guidance_scale_lora * (
                noise_pred_est_text - noise_pred_est_uncond
        )

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)

        grad = w * (noise_pred_pretrain - noise_pred_est)
        return grad

    def train_step(self, text_embeddings, pred_rgb, camera_condition,
                   guidance_scale=10,
                   as_latents=False,
                   t_range=(0.02, 0.98)):

        B = pred_rgb.size(0)
        latents = self.get_latents(pred_rgb, rgb_as_latents=as_latents)

        grad = self.compute_grad_vsd(latents, text_embeddings, camera_condition, t_range, guidance_scale=guidance_scale)
        grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()
        loss_vsd = 0.5 * F.mse_loss(latents, target, reduction="sum") / B

        loss_lora = self.train_lora(latents, self.text_embeddings_invd, camera_condition)

        return loss_vsd + loss_lora
