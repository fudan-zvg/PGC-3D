import os
import glob
import tqdm
import math
import imageio
import random
import warnings
import tensorboardX
import time

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist

from rich.console import Console
from torch_ema import ExponentialMovingAverage
from collections import defaultdict
from guidance.guidance_utils import precompute_prior, time_prioritize


class Trainer(object):
    def __init__(self,
                 name,  # name of this experiment
                 opt,  # extra conf
                 model,  # network
                 guidance,  # guidance network
                 guidance_o=None,  # other guidance
                 criterion=None,  # loss function, if None, assume inline implementation in train_step
                 optimizer=None,  # optimizer
                 ema_decay=None,  # if use EMA, set the decay
                 lr_scheduler=None,  # scheduler
                 metrics=None,  # metrics for evaluation
                 local_rank=0,  # which GPU am I
                 world_size=1,  # total num of GPUs
                 device=None,  # device to use, usually setting to None is OK. (auto choose device)
                 mute=False,  # whether to mute all print
                 eval_interval=1,  # eval once every $ epoch
                 max_keep_ckpt=1,  # max num of saved ckpts in disk
                 workspace='workspace',  # workspace to save logs & ckpts
                 best_mode='min',  # the smaller/larger result, the better
                 use_loss_as_metric=True,  # use loss as the first metric
                 report_metric_at_train=False,  # also report metrics at training
                 use_checkpoint="latest",  # which ckpt to use at init time
                 use_tensorboardX=False,  # whether to use tensorboard for logging
                 scheduler_update_every_step=True,  # whether to call scheduler.step() after every train step
                 ):
        self.stage = "fine"
        self.name = f'{self.stage}_' + name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(
            'cuda:{}'.format(local_rank) if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.fp16 = opt.fp16

        self.model = model.to(device)

        # guide model
        self.guidance = guidance
        self.guidance_o = guidance_o
        if self.guidance_o is not None:
            for p in self.guidance_o.parameters():
                p.requires_grad = False

        # text prompt
        if self.guidance is not None:
            if not self.opt.vsd:
                for p in self.guidance.parameters():
                    p.requires_grad = False

            self.prepare_text_embeddings()
        else:
            self.text_z = None

        # dreamtime
        if self.opt.dreamtime:
            self.time_prior, _ = precompute_prior(max_t=int(self.opt.init_time * 1000))

        # reference image
        self.ref_images = None
        self.res = self.opt.res_fine
        if self.opt.ref_path:
            if os.path.isdir(self.opt.ref_path):
                files = os.listdir(self.opt.ref_path)
                files.sort()
                files = [os.path.join(self.opt.ref_path, file) for file in files]
            else:
                files = [self.opt.ref_path]

            imgs = []
            masks = []
            for file in files:
                img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
                if img.shape[-1] == 4:
                    mask = img[..., -1:] / 255
                else:
                    mask = np.ones_like(img[..., 0:1])
                img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB) / 255
                img = torch.tensor(img, dtype=torch.float32, device=self.device)[None, ...]
                mask = torch.tensor(mask, dtype=torch.float32, device=self.device)[None, ...]
                imgs.append(img)
                masks.append(mask)
            ref_images = torch.cat(imgs, dim=0).permute(0, 3, 1, 2)
            ref_mask = torch.cat(masks, dim=0).permute(0, 3, 1, 2)
            self.ref_images = F.interpolate(ref_images, (self.res, self.res), mode='bilinear', align_corners=False)
            self.ref_mask = F.interpolate(ref_mask, (self.res, self.res), mode='bilinear', align_corners=False)

            if self.opt.zero123:
                mask = self.ref_mask.expand(self.ref_mask.shape[0], 3, self.res, self.res)
                ref_imgs = self.ref_images * mask + (1 - mask) * 0.8
                image_256 = F.interpolate(ref_imgs, (256, 256), mode='bilinear', align_corners=False)
                self.ref_z = self.guidance_o.get_img_embeds(image_256[0:1])

        # grad scale
        if self.opt.zero123:
            self.grad_scale_sd = 0.01
            self.grad_scale_rgb = 1.0
        else:
            self.grad_scale_sd = 1.0
            self.grad_scale_rgb = 1.0

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        params = self.model.get_params(opt.lr_fine)
        if self.opt.vsd:
            params.append(
                {'params': self.guidance.parameters(), 'lr': opt.lr_fine[1] / 10}
            )
        if optimizer is None:
            self.optimizer = optim.Adam(params, lr=0.001, weight_decay=5e-4)  # naive adam
        else:
            self.optimizer = optimizer(params)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1)  # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(init_scale=2. ** 10, enabled=self.fp16)

        if self.world_size > 1:
            process_group = torch.distributed.new_group(list(range(dist.get_world_size())))
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model, process_group)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                   device_ids=[local_rank],
                                                                   output_device=local_rank,
                                                                   find_unused_parameters=True)
            self.module = self.model.module
        else:
            self.module = self.model

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if metrics is None or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(f'[INFO] Trainer: {self.time_stamp} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
        self.log(f'[INFO] opt: {self.opt}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else:  # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    # calculate the text embs.
    def prepare_text_embeddings(self):

        if self.opt.text is None:
            self.log(f"[WARN] text prompt is not provided.")
            self.text_z = None
            return

        if not self.opt.dir_text:
            self.text_z = self.guidance.get_text_embeds([self.opt.text], [self.opt.negative])
        else:
            self.text_z = []
            for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom']:
                # construct dir-encoded text
                # text = f"{self.opt.text}"
                text = f"{self.opt.text}, {d} view"
                # text = f"{d} view of {self.opt.text}"
                negative_text = f"{self.opt.negative}"

                # explicit negative dir-encoded text
                if self.opt.suppress_face and d != 'front':
                    if negative_text != '': negative_text += ', '
                    negative_text += "face"

                text_z = self.guidance.get_text_embeds([text], [negative_text])

                # text_z_part = [text_z]
                # for part in ['head', 'hand']:
                #     text = f"The {d} view of the {part} of {self.opt.text}, {d} view"
                #     text_z_part.append(self.guidance.get_text_embeds([text], [negative_text]))
                # text_z_part = torch.stack(text_z_part, dim=0)

                self.text_z.append(text_z)
            if 'xl' in self.opt.guidance:
                text_z_tmp = []
                for i in range(4):
                    text_z_tmp.append(torch.stack([emb[i] for emb in self.text_z], dim=0))
                self.text_z = text_z_tmp
            else:
                self.text_z = torch.stack(self.text_z, dim=0)

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                # print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()  # write immediately to file

    ### ------------------------------

    def train_step(self, data):
        B, N = data['rays_d'].shape[:2]  # [B, N, 3]
        H, W = data['H'], data['W']

        smooth = False
        if (self.opt.lambda_smooth > 0) and (
                self.global_step > self.opt.iters - self.opt.smooth_iters):
            smooth = True

        if self.global_step < self.opt.albedo_iters:
            shading = 'albedo'
            ambient_ratio = 1.0
        else:
            rand = random.random()
            if rand > 0.8:
                shading = 'albedo'
                ambient_ratio = 1.0
            else:
                shading = 'lambertian'
                ambient_ratio = 0.1

        pred_rgb, outputs = self.model(data,
                                       ratio=ambient_ratio,
                                       shading=shading,
                                       smooth=smooth)
        pred_rgb = pred_rgb.permute(0, 3, 1, 2)
        normals = (-outputs['normals'] + 1) / 2
        normals = normals.permute(0, 3, 1, 2)[:B]

        # time
        t_dreamtime = None
        t_bgt = None
        if self.opt.dreamtime:
            step_ratio = (self.global_step - 1) / self.opt.iters
            t = time_prioritize(step_ratio, self.time_prior)
            t_dreamtime = int(t)
        if self.opt.bgt:
            max_t = self.opt.init_time
            ratio = math.sqrt((self.global_step - 1) / float(self.opt.iters))
            t_bgt = max_t - (max_t - 0.25) * ratio

        loss = torch.tensor(0., device=self.device, requires_grad=True)

        # reference image
        if self.ref_images is not None:
            pred_ref = pred_rgb[B:]
            pred_rgb = pred_rgb[:B]
            mask = self.ref_mask.expand(pred_ref.shape[0], 3, self.res, self.res)
            loss = loss + torch.square(pred_ref - self.ref_images)[mask > 0].sum() * self.opt.lambda_ref

            if 'clip' in self.opt.guidance_o:
                ref_imgs = self.ref_images * mask + (1 - mask)
                ref_z = self.guidance_o.get_image_embeds(ref_imgs)
                loss = loss + self.guidance_o.train_step(ref_z, pred_ref)

            if self.opt.zero123:
                dT = data['dT']
                if self.opt.bgt:
                    loss = loss + self.guidance_o.train_step_BGT(self.ref_z, pred_rgb, dT, t=t_bgt)
                else:
                    loss = loss + self.guidance_o.train_step(self.ref_z, pred_rgb, dT,
                                                             t_range=[0.02, 0.5])

        # text embeddings
        if self.opt.dir_text:
            dirs = data['dir']  # [B,]
            if 'xl' in self.opt.guidance:
                prompt_embeds = self.text_z[0][dirs].transpose(0, 1).flatten(0, 1)
                negative_prompt_embeds = self.text_z[1][dirs].transpose(0, 1).flatten(0, 1)
                pooled_prompt_embeds = self.text_z[2][dirs].transpose(0, 1).flatten(0, 1)
                negative_pooled_prompt_embeds = self.text_z[3][dirs].transpose(0, 1).flatten(0, 1)
                text_z = (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
            else:
                text_z = self.text_z[dirs].transpose(0, 1).flatten(0, 1)
        else:
            text_z = self.text_z.repeat_interleave(B, dim=0)

        # condition image
        use_control = 'controlnet' in self.opt.guidance
        cond_imgs = []
        for control_type in self.opt.control_type:
            if 'depth' in control_type:
                mask = outputs['mask'][:B]
                depth = outputs['depth'][:B]
                bg_depth = depth.max() * 1.2
                depth = depth * mask + bg_depth * (1 - mask)
                cond_img = depth.permute(0, 3, 1, 2)
            elif 'normal' in control_type:
                cond_img = 2.0 * normals - 1.0
            elif 'tile' in control_type:
                cond_img = 2.0 * pred_rgb - 1.0
            elif 'shuffle' in control_type and self.ref_images is not None:
                cond_img = 2.0 * self.ref_images[0:1] - 1.0
                cond_img = cond_img.expand(B, 3, self.res, self.res)
            else:
                cond_img = None

            cond_img = cond_img.detach()
            cond_imgs.append(cond_img)

        # normal guidance
        if not self.opt.no_normal or self.opt.only_normal:
            if use_control:
                loss = loss + self.guidance.train_step(text_z, normals, cond_imgs,
                                                       t_range=[0.02, 0.5],
                                                       guidance_scale=50)
            else:
                loss = loss + self.guidance.train_step(text_z, normals,
                                                       t_range=[0.02, 0.5],
                                                       guidance_scale=50)

        # texture guidance
        if not self.opt.only_normal:
            if self.opt.bgt and not self.opt.zero123:
                loss = loss + self.guidance.train_step_BGT(text_z, pred_rgb,
                                                           guidance_scale=self.opt.guidance_scale,
                                                           t=t_bgt)
            elif use_control:
                latents = pred_rgb if self.opt.latent else None
                loss = loss + self.guidance.train_step(text_z, pred_rgb, cond_imgs,
                                                       guidance_scale=self.opt.guidance_scale,
                                                       grad_scale=self.grad_scale_sd,
                                                       t_range=[0.02, 0.5], t=t_dreamtime,
                                                       latents=latents)
            elif self.opt.vsd:
                loss = loss + self.guidance.train_step(text_z, pred_rgb, data['poses'],
                                                       guidance_scale=self.opt.guidance_scale,
                                                       t_range=[0.02, 0.5],
                                                       as_latents=self.opt.latent)
            else:
                latents = pred_rgb if self.opt.latent else None
                loss = loss + self.guidance.train_step(text_z, pred_rgb,
                                                       guidance_scale=self.opt.guidance_scale,
                                                       grad_scale=self.grad_scale_sd,
                                                       t_range=[0.02, 0.5], t=t_dreamtime,
                                                       latents=latents)

        # other loss
        if self.opt.lambda_smooth > 0 and 'loss_smooth' in outputs:
            loss_smooth = outputs['loss_smooth']
            loss = loss + self.opt.lambda_smooth * loss_smooth

        if self.opt.pbr:
            warm_up_iters = self.opt.iters / 10
            loss = loss + outputs['reg_kd_smooth'] * 0.03 * min(1.0, self.global_step / warm_up_iters)
            # loss = loss + outputs['reg_ks_smooth'] * 0.03 * min(1.0, self.global_step / warm_up_iters)
            # loss = loss + outputs['reg_vis'] * 0.001 * min(1.0, self.global_step / warm_up_iters)
            # loss = loss + outputs['reg_lgt'] * 0.01
            # loss = loss + (outputs['spec_col'] ** 2).mean(0).sum() * 1e-5

        if self.opt.reconstruction:
            loss = loss + torch.square(pred_rgb - data['images']).sum() / B * 0.1

        if self.opt.lambda_saturation > 0:
            img = pred_rgb * 0.5 + 0.5
            mask = outputs['mask']
            saturation = (img.max(1)[0] - img.min(1)[0] ) / (img.max(1)[0] + 1e-7)
            saturation[img.max(1)[0] == 0] = 0
            mean_sat = saturation[mask[..., 0] > 0].mean()
            if mean_sat > 0.5:
                loss = loss + torch.square(mean_sat) * self.opt.lambda_saturation

        return pred_rgb, normals, loss

    def eval_step(self, data):
        rays_d = data['rays_d']  # [B, N, 3]

        B, N = rays_d.shape[:2]
        H, W = data['H'], data['W']

        rays_d = data['rays_d']  # [B, N, 3]

        B, N = rays_d.shape[:2]
        H, W = data['H'], data['W']

        shading = 'albedo'
        ambient_ratio = 1.0

        pred_rgb, outputs = self.model(data,
                                       ratio=ambient_ratio,
                                       shading=shading,
                                       eval=True)
        pred_depth = outputs['depth'].reshape(B, H, W)
        pred_normal = outputs['normals_vis']
        if self.opt.latent:
            res = 128 if 'xl' in self.opt.guidance else 64
            latents = F.interpolate(
                pred_rgb.permute(0, 3, 1, 2), (res, res), mode="bilinear", align_corners=False
            )
            pred_rgb = self.guidance.decode_latents(latents).permute(0, 2, 3, 1)

        loss = torch.tensor(0., device=self.device)
        if self.opt.lambda_smooth > 0 and 'loss_smooth' in outputs:
            loss_smooth = outputs['loss_smooth']
            loss = loss + self.opt.lambda_smooth * loss_smooth

        return pred_rgb, pred_depth, pred_normal, loss

    def test_step(self, data):
        rays_d = data['rays_d']  # [B, N, 3]

        B, N = rays_d.shape[:2]
        H, W = data['H'], data['W']

        shading = 'albedo'
        ambient_ratio = 1.0

        pred_rgb, outputs = self.model(data,
                                       ratio=ambient_ratio,
                                       shading=shading,
                                       eval=True)
        pred_depth = outputs['depth'].reshape(B, H, W)
        pred_normal = outputs['normals_vis']

        if 'xl' in self.opt.guidance and self.opt.use_refiner:
            latents = F.interpolate(
                pred_rgb.permute(0, 3, 1, 2), (128, 128), mode="bilinear", align_corners=False
            )
            for j in range(1):
                latents = self.guidance.refiner(prompt=self.opt.text,
                                                image=latents,
                                                output_type="latent").images

            pred_rgb = self.guidance.decode_latents(latents, refiner_vae=True).permute(0, 2, 3, 1)
        elif self.opt.latent:
            res = 128 if 'xl' in self.opt.guidance else 64
            latents = F.interpolate(
                pred_rgb.permute(0, 3, 1, 2), (res, res), mode="bilinear", align_corners=False
            )
            pred_rgb = self.guidance.decode_latents(latents).permute(0, 2, 3, 1)

        return pred_rgb, pred_depth, pred_normal

    @torch.cuda.amp.autocast(enabled=False)
    def save_mesh(self, save_path=None, name=None, mesh_type=None):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'mesh')
            if self.opt.mesh_name:
                save_path = os.path.join(self.workspace, 'mesh', self.opt.mesh_name)

        if name is None:
            name = self.opt.mesh_name

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(save_path, exist_ok=True)

        if mesh_type is None:
            mesh_type = self.opt.mesh_type
        self.module.export_mesh(save_path, name=name, type=mesh_type, SD=self.guidance)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs, refine_epochs):

        assert self.text_z is not None, 'Training must provide a text prompt!'

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        self.max_epochs = max_epochs

        start_t = time.time()

        for epoch in range(self.epoch + 1, max_epochs + refine_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.model.train()
                # if self.local_rank == 0:
                #     # path = os.path.join(self.workspace, 'mesh', f'{self.name}_ep{self.epoch:04d}')
                #     # os.makedirs(path, exist_ok=True)
                #     # self.save_mesh(path, mesh_type='vertex_color')
                #     self.save_checkpoint(full=False, best=True)
            # dist.barrier()

        end_t = time.time()

        self.log(f"[INFO] training takes {(end_t - start_t) / 60:.4f} minutes.")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.model.eval()
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                         bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        all_preds = []
        all_preds_depth = []
        all_preds_normal = []
        all_index = []
        all_poses = []

        with torch.no_grad():

            for i, data in enumerate(loader):
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, preds_normal = self.test_step(data)
                preds = preds.contiguous()
                preds_depth = preds_depth.contiguous()
                preds_normal = preds_normal.contiguous()

                if self.world_size > 0:
                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in
                                  range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in
                                        range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    preds_normal_list = [torch.zeros_like(preds_normal).to(self.device) for _ in
                                         range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_normal_list, preds_normal)
                    preds_normal = torch.cat(preds_normal_list, dim=0)

                    index_list = [torch.zeros_like(data['index']).to(self.device) for _ in
                                  range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(index_list, data['index'])
                    index = torch.cat(index_list, dim=0)

                    poses_list = [torch.zeros_like(data['poses']).to(self.device) for _ in
                                  range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(poses_list, data['poses'])
                    poses = torch.cat(poses_list, dim=0)

                pred = preds.detach().cpu().numpy()
                pred = np.clip(pred * 255, 0, 255).astype(np.uint8)

                depth_min = torch.amin(preds_depth, dim=[1, 2], keepdim=True)
                depth_max = torch.amax(preds_depth, dim=[1, 2], keepdim=True)
                preds_depth = (preds_depth - depth_min) / (depth_max - depth_min)
                pred_depth = preds_depth.detach().cpu().numpy()
                pred_depth = np.clip(pred_depth * 255, 0, 255).astype(np.uint8)

                pred_normal = preds_normal.detach().cpu().numpy()
                pred_normal = (pred_normal * 255).astype(np.uint8)

                pose = poses.detach().cpu().numpy()

                all_preds.append(pred)
                all_preds_depth.append(pred_depth)
                all_preds_normal.append(pred_normal)
                all_index.append(index.detach().cpu().numpy())
                all_poses.append(pose)
                pbar.update(loader.batch_size)

        all_index = np.concatenate(all_index, axis=0)
        all_index = np.argsort(all_index)
        all_preds = np.concatenate(all_preds, axis=0)[all_index]
        all_preds_depth = np.concatenate(all_preds_depth, axis=0)[all_index]
        all_preds_normal = np.concatenate(all_preds_normal, axis=0)[all_index]
        all_poses = np.concatenate(all_poses, axis=0)[all_index]
        imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=25, quality=8,
                         macro_block_size=1)
        imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=25, quality=8,
                         macro_block_size=1)
        imageio.mimwrite(os.path.join(save_path, f'{name}_normal.mp4'), all_preds_normal, fps=25, quality=8,
                         macro_block_size=1)
        for i in range(all_index.shape[0]):
            os.makedirs(os.path.join(save_path, 'rgb'), exist_ok=True)
            os.makedirs(os.path.join(save_path, 'depth'), exist_ok=True)
            os.makedirs(os.path.join(save_path, 'normal'), exist_ok=True)
            cv2.imwrite(os.path.join(save_path, 'rgb', f'{i:04d}.png'),
                        cv2.cvtColor(all_preds[i], cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_path, 'depth', f'{i:04d}.png'),
                        all_preds_depth[i])
            cv2.imwrite(os.path.join(save_path, 'normal', f'{i:04d}.png'),
                        cv2.cvtColor(all_preds_normal[i], cv2.COLOR_RGB2BGR))

        np.save(os.path.join(save_path, f'poses.npy'), all_poses)

        self.log(f"==> Finished Test.")

    def train_one_epoch(self, loader):
        self.log(
            f"==> Start Training {self.workspace} Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:

            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                pred_rgbs, normals, loss = self.train_step(data)

            if self.opt.grad_clip_rgb >= 0:
                def _hook(grad):
                    grad = grad * self.grad_scale_rgb

                    if self.opt.fp16:
                        # correctly handle the scale
                        grad_scale = self.scaler._get_scale_async()
                        clip_value = grad_scale * self.opt.grad_clip_rgb
                    else:
                        grad_scale = 1.0
                        clip_value = self.opt.grad_clip_rgb

                    if self.opt.grad_suppress_type == 0:  # pwclip
                        ratio = 1. / grad.abs() * clip_value
                        ratio[ratio > 1.0] = 1.0
                        grad_ = grad * torch.amin(ratio, dim=[1], keepdim=True)
                    elif self.opt.grad_suppress_type == 1:  # clip
                        grad_ = grad.clamp(-clip_value, clip_value)
                    elif self.opt.grad_suppress_type == 2:  # global scale
                        grad_ = grad / grad.abs().max() * clip_value
                    elif self.opt.grad_suppress_type == 3:  # sigmoid
                        grad_ = (torch.sigmoid(grad) - 0.5) * clip_value
                    elif self.opt.grad_suppress_type == 4:  # norm
                        grad_norm = grad.abs()  # torch.amax(grad.abs(), dim=[1], keepdim=True)
                        grad_ = clip_value * (grad / (grad_norm + clip_value))
                    elif self.opt.grad_suppress_type == 5:  # norm
                        grad_norm = torch.amax(grad.abs(), dim=[1], keepdim=True)
                        grad_ = clip_value * (grad / (grad_norm + clip_value))
                    else:
                        grad_ = grad
                    return grad_

                pred_rgbs.register_hook(_hook)
                if not self.opt.no_normal:
                    normals.register_hook(_hook)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.opt.grad_clip >= 0:
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.opt.grad_clip)
            self.scaler.step(self.optimizer)
            # optimizer_state = self.scaler._per_optimizer_states[id(self.optimizer)]
            # if sum(v.item() for v in optimizer_state["found_inf_per_device"].values()):
            #     print("skip")
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate {self.workspace} at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            for data in loader:
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, preds_normal, loss = self.eval_step(data)
                preds = preds.contiguous()
                preds_depth = preds_depth.contiguous()
                preds_normal = preds_normal.contiguous()

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size

                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in
                                  range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in
                                        range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    preds_normal_list = [torch.zeros_like(preds_normal).to(self.device) for _ in
                                         range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_normal_list, preds_normal)
                    preds_normal = torch.cat(preds_normal_list, dim=0)

                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:
                    # save images
                    save_path = os.path.join(self.workspace, 'validation')
                    save_path_rgb = os.path.join(
                        save_path,
                        f'{name}_{self.local_step:04d}_rgb.png'
                    )
                    save_path_depth = os.path.join(
                        save_path,
                        f'{name}_{self.local_step:04d}_depth.png'
                    )
                    save_path_normal = os.path.join(
                        save_path,
                        f'{name}_{self.local_step:04d}_normal.png'
                    )

                    # self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(save_path, exist_ok=True)

                    pred = preds[0].detach().cpu().numpy()
                    pred = np.clip(pred * 255, 0, 255).astype(np.uint8)

                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    pred_depth = (pred_depth * 255).astype(np.uint8)

                    pred_normal = preds_normal[0].detach().cpu().numpy()
                    pred_normal = np.clip(pred_normal * 255, 0, 255).astype(np.uint8)

                    cv2.imwrite(save_path_rgb, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    if not self.opt.no_normal or self.opt.only_normal:
                        cv2.imwrite(save_path_normal, pred_normal)
                    # cv2.imwrite(save_path_depth, pred_depth)

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                    pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            self.stats["results"].append(average_loss)  # if no metric, choose best by min loss

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        state['model'] = self.module.state_dict()
        file_path = f"{name}.pth"
        self.stats["checkpoints"].append(file_path)
        if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
            old_ckpt = os.path.join(self.ckpt_path, self.stats["checkpoints"].pop(0))
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)

        torch.save(state, os.path.join(self.ckpt_path, file_path))

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.stage}*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.module.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.module.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and 'ema' in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")

    ### --------------------------------
    def train_initsdf(self, query_points, target_sdf, target_texture=None, max_steps=500):
        start_t = time.time()
        losses = []
        lr = 1e-3
        self.optimizer.param_groups[0]['lr'] = lr * 10
        self.optimizer.param_groups[1]['lr'] = lr
        self.optimizer.param_groups[2]['lr'] = lr * 10
        self.optimizer.param_groups[3]['lr'] = lr
        if self.opt.pbr:
            self.optimizer.param_groups[2]['lr'] = lr

        for i in range(1, max_steps + 1):
            self.optimizer.zero_grad()
            pred_sdf, pred_texture = self.model(0, only_points=True, query_points=query_points)
            loss = ((pred_sdf - target_sdf) ** 2).mean()
            if target_texture is not None:
                loss = loss + ((pred_texture - target_texture) ** 2).sum(-1).mean()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            if self.local_rank == 0 and i % 100 == 0:
                mloss = sum(losses[-100:]) / 100
                self.log(f"step{i} loss:{mloss:.10f}")

        end_t = time.time()

        self.log(f"[INFO] init sdf takes {(end_t - start_t) / 60:.4f} minutes.")

        # reset optimizer
        self.reset_optimizer()
        self.save_checkpoint(full=False, best=True)

    def reset_optimizer(self):
        self.optimizer.state = defaultdict(dict)
        self.optimizer.param_groups[0]['lr'] = self.opt.lr_fine[0] * 10
        self.optimizer.param_groups[1]['lr'] = self.opt.lr_fine[0]
        if self.opt.pbr:
            self.optimizer.param_groups[2]['lr'] = self.opt.lr_fine[1]
            self.optimizer.param_groups[3]['lr'] = self.opt.lr_fine[1] * 0.1
        else:
            self.optimizer.param_groups[2]['lr'] = self.opt.lr_fine[1] * 10
            self.optimizer.param_groups[3]['lr'] = self.opt.lr_fine[1]

    def train_initsdf_fixmesh(self, query_points, target_texture, max_steps=500):
        start_t = time.time()
        losses = []

        for i in range(1, max_steps + 1):
            self.optimizer.zero_grad()
            pred_texture = self.module.get_texture(query_points)[..., :3]
            loss = ((pred_texture - target_texture) ** 2).sum(-1).mean()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            if self.local_rank == 0 and i % 100 == 0:
                mloss = sum(losses[-100:]) / 100
                self.log(f"step{i} loss:{mloss:.10f}")

        end_t = time.time()

        self.log(f"[INFO] init sdf takes {(end_t - start_t) / 60:.4f} minutes.")

        # reset optimizer
        self.optimizer.state = defaultdict(dict)
        self.save_checkpoint(full=False, best=True)
