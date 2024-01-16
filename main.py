import argparse

from nerf.provider import RastDataset
from nerf.utils import *
import torch.distributed as dist
import torch.multiprocessing as mp


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default=None, help="text prompt")
    parser.add_argument('--negative', default='', type=str, help="negative text prompt")
    parser.add_argument('-O0', action='store_true', help="grad clip")
    parser.add_argument('-O1', action='store_true', help="deformable tet grid")
    parser.add_argument('-O2', action='store_true', help="fix mesh")
    parser.add_argument('-O3', action='store_true', help="image to 3d")
    parser.add_argument('-O4', action='store_true', help="SDXL")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--test_mesh', action='store_true', help="test on mesh")
    parser.add_argument('--save_mesh', action='store_true', help="export an obj mesh with texture")
    parser.add_argument('--eval_interval', type=int, default=1, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--guidance', type=str, default='stable-diffusion')
    parser.add_argument('--guidance_o', type=str, default='stable-diffusion')
    parser.add_argument('--guidance_scale', type=float, default=100)
    parser.add_argument('--control_type', type=str, nargs='*', default=['depth'])
    parser.add_argument('--control_scale', type=float, nargs='*', default=[1.0])
    parser.add_argument('--sd_version', type=str, default='2.1')
    parser.add_argument('--load_finetuned', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mesh_name', type=str, default='')
    parser.add_argument('--mesh_type', type=str, default='')
    parser.add_argument('--from_coarse', action='store_true')
    parser.add_argument('--mesh_path', type=str, default='')
    parser.add_argument('--canonicalize', action='store_true')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--scheduler', type=str, default='ddpm')
    parser.add_argument('--use_refiner', action='store_true')

    # setting
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--no_normal', action='store_true', help="normal map as input to sd")
    parser.add_argument('--no_MLP', action='store_true', help="predict SDF using MLP")
    parser.add_argument('--only_normal', action='store_true', help="only normal map as input to sd")

    parser.add_argument('--zero123', action='store_true', help="use zero123")
    parser.add_argument('--left', action='store_true', help="reference image toward left")
    parser.add_argument('--ref_path', type=str, default='', help="path to reference image")
    parser.add_argument('--lambda_ref', type=float, default=0.1)

    parser.add_argument('--dreamtime', action='store_true', help="an annealing time schedule proposed by DreamTime")
    parser.add_argument('--init_time', type=float, default=0.681, help="initial time")

    parser.add_argument('--pbr', action='store_true', help="use pbr")
    parser.add_argument('--no_perturbed_nrm', action='store_true', help="no perturbed normal")
    parser.add_argument('--ks_min', type=float, nargs='*', default=[0.0, 0.1, 0.0])
    parser.add_argument('--ks_max', type=float, nargs='*', default=[0.0, 0.98, 0.0])

    parser.add_argument('--fix_mesh', action='store_true', help="no dmtet")
    parser.add_argument('--init_texture', action='store_true', help="initialize texture if mesh has vertex color")
    parser.add_argument('--pos_perturb', action='store_true', help="perturb input positions during training")

    parser.add_argument('--latent', action='store_true', help="latent nerf mode")
    parser.add_argument('--vsd', action='store_true', help="proposed by ProlificDreamer")
    parser.add_argument('--bgt', action='store_true', help="proposed by HiFA")
    parser.add_argument('--csd', action='store_true', help="Classifier Score Distillation")
    parser.add_argument('--ssd', action='store_true', help="Stable Score Distillation")
    parser.add_argument('--ssd_M', type=int, default=100)
    parser.add_argument('--cfg_rescale', action='store_true', help="suggested by AvatarStudio")

    parser.add_argument('--bg_color', type=float, default=1.0)
    parser.add_argument('--bg_scenery', action='store_true', help="use a scenery as background")

    parser.add_argument('--reconstruction', action='store_true')
    parser.add_argument('--data_root', type=str, default='')

    # special loss
    parser.add_argument('--lambda_entropy', type=float, default=1e-4, help="loss scale for alpha entropy")
    parser.add_argument('--lambda_opacity', type=float, default=0, help="loss scale for alpha value")
    parser.add_argument('--lambda_orient', type=float, default=1e-2, help="loss scale for orientation")
    parser.add_argument('--lambda_smooth', type=float, default=2e6, help="loss scale for surface smoothness")
    parser.add_argument('--lambda_mask', type=float, default=0, help="loss scale for alpha entropy")
    parser.add_argument('--lambda_saturation', type=float, default=0, help="loss scale for saturation")

    ### training options
    parser.add_argument('--iters', type=int, default=2000, help="training iters")
    parser.add_argument('--warmup_iters', type=int, default=0, help="normal+mask as latent")
    parser.add_argument('--smooth_iters', type=int, default=1000, help="smooth loss")
    parser.add_argument('--refine_iters', type=int, default=0, help="refine mesh training iters")
    parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
    parser.add_argument('--lr_fine', type=float, nargs='*', default=[1e-5, 1e-3],
                        help="learning rate for sdf and texture")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--ckpt_fine', type=str, default='scratch')
    parser.add_argument('--albedo', action='store_true',
                        help="only use albedo shading to train, overrides --albedo_iters")
    parser.add_argument('--albedo_iters', type=int, default=0, help="training iters that only use albedo shading")
    parser.add_argument('--uniform_sphere_rate', type=float, default=0,
                        help="likelihood of sampling camera location uniformly on the sphere surface area")
    parser.add_argument('--weight_choice', type=int, default=0, help="choice for w(t)")
    parser.add_argument('--grad_clip', type=float, default=-1,
                        help="clip grad of all grad to this limit, negative value disables it")
    parser.add_argument('--grad_clip_rgb', type=float, default=-1,
                        help="clip grad of rgb space grad to this limit, negative value disables it")
    parser.add_argument('--grad_clip_latent', type=float, default=-1,
                        help="clip grad of latent space grad to this limit, negative value disables it")
    parser.add_argument('--grad_suppress_type', type=int, default=0)

    # gpus
    parser.add_argument('--local_rank', type=int, default=0, help="node rank for distributed training")
    parser.add_argument('--gpus', type=str, default="1", help="devices: '0,1,2,3' ")
    parser.add_argument('--bs', type=str, default=4, help="per gpu batch size")
    parser.add_argument('--port', type=int, default=6666, help="port, arbitrary number in 0~65536")

    # rendering resolution in training, decrease this if CUDA OOM.
    parser.add_argument('--w', type=int, default=64, help="render width for NeRF in training")
    parser.add_argument('--h', type=int, default=64, help="render height for NeRF in training")
    parser.add_argument('--jitter_pose', action='store_true', help="add jitters to the randomly sampled camera poses")
    parser.add_argument('--res_fine', type=int, default=512, help="image resolution in fine stage")
    parser.add_argument('--tet_res', type=int, default=256, help="resolution for tetrahedron grid")

    ### dataset options
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--dt_gamma', type=float, default=0,
                        help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.1, help="minimum near distance for camera")
    parser.add_argument('--radius_range', type=float, nargs='*', default=[2.0, 2.5],
                        help="training camera radius range")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[40, 70], help="training camera fovy range")
    parser.add_argument('--focal_range', type=float, nargs='*', default=[0.7, 1.35], help="training camera focal range")
    parser.add_argument('--focal_range_fine', type=float, nargs='*', default=[1.0, 1.6],
                        help="training camera focal range in fine stage")
    parser.add_argument('--dir_text', action='store_true',
                        help="direction-encode the text prompt, by appending front/side/back/overhead view")
    parser.add_argument('--suppress_face', action='store_true', help="also use negative dir text prompt.")
    parser.add_argument('--angle_overhead', type=float, default=30, help="[0, angle_overhead] is the overhead region")
    parser.add_argument('--angle_front', type=float, default=90,
                        help="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")

    parser.add_argument('--n_targets', type=int, default=0)
    parser.add_argument('--local_prob', type=float, default=0.5)
    parser.add_argument('--ls_all', action='store_true', help="local sample in all stage, default skip s1")
    parser.add_argument('--single_stage', action='store_true', help="no increasing focal")

    opt = parser.parse_args()
    return opt


def main_worker(local_rank, nprocs, opt):
    opt.local_rank = local_rank
    torch.cuda.set_device(opt.local_rank)

    dist.init_process_group(backend="nccl",
                            init_method=f'tcp://127.0.0.1:{opt.port}',
                            world_size=nprocs,
                            rank=local_rank)
    opt.world_size = dist.get_world_size()
    opt.global_rank = dist.get_rank()
    opt.bs = int(opt.bs)
    seed_everything(opt.seed + opt.global_rank)

    device = torch.device('cuda', opt.local_rank)

    if opt.reconstruction:
        from nerf.provider import RecDataset as TrainDataset
    else:
        TrainDataset = RastDataset

    if opt.test:
        guidance = None  # no need to load guidance model at test,
        if 'xl' in opt.guidance:
            from guidance.sdxl import StableDiffusionXL
            guidance = StableDiffusionXL(device, opt, use_refiner=opt.use_refiner)

        from nerf.fine.trainer import Trainer as FineTrainer
        from nerf.fine.rasterizer import Rasterizer_MLP
        from nerf.fine.rasterizer_pbr import Rasterizer_pbr
        from nerf.fine.rasterizer_mesh import Rasterizer_mesh

        if opt.mesh_path:
            mesh_path = opt.mesh_path
        else:
            mesh_path = os.path.join(opt.workspace, 'mesh', 'fine_mesh.obj')

        if opt.fix_mesh:
            model = Rasterizer_mesh(opt=opt, mesh_path=mesh_path)
        elif opt.pbr:
            model = Rasterizer_pbr(opt=opt, tet_res=opt.tet_res, mesh_path=mesh_path)
        else:
            model = Rasterizer_MLP(opt=opt, tet_res=opt.tet_res, mesh_path=mesh_path)

        test_loader = RastDataset(opt, device=device, type='test',
                                  H=512, W=512, size=100).dataloader()

        trainer = FineTrainer('df', opt, model, guidance, device=device,
                              workspace=opt.workspace, use_checkpoint=opt.ckpt,
                              local_rank=opt.local_rank, world_size=opt.world_size)
        trainer.test(test_loader)

        if opt.save_mesh:
            trainer.save_mesh()

    else:

        if ('xl' in opt.guidance) and ('controlnet' in opt.guidance):
            from guidance.sdxl_controlnet import SDXLControlNet
            guidance = SDXLControlNet(device, opt)
        elif 'controlnet' in opt.guidance:
            from guidance.sdcontrolnet import SDControlNet
            guidance = SDControlNet(device, opt)
        elif 'xl' in opt.guidance:
            if opt.vsd:
                from guidance.sdxl_vsd import StableDiffusionXLvsd
                guidance = StableDiffusionXLvsd(device, opt, use_refiner=opt.use_refiner)
            else:
                from guidance.sdxl import StableDiffusionXL
                guidance = StableDiffusionXL(device, opt, use_refiner=opt.use_refiner)
        else:
            from guidance.sd import StableDiffusion
            guidance = StableDiffusion(device, opt)

        guidance_others = None
        if opt.zero123:
            from guidance.zero123_utils import Zero123
            guidance_others = Zero123(device)
        if 'clip' in opt.guidance_o:
            from guidance.clip_utils import CLIP
            guidance_others = CLIP(device)

        from nerf.fine.trainer import Trainer as FineTrainer
        from nerf.fine.rasterizer import Rasterizer_MLP
        from nerf.fine.rasterizer_mesh import Rasterizer_mesh
        from nerf.fine.rasterizer_pbr import Rasterizer_pbr

        if opt.mesh_path:
            mesh_path = opt.mesh_path
        else:
            mesh_path = os.path.join(opt.workspace, 'mesh', 'initfine_mesh_vc.obj')

        if opt.fix_mesh:
            model = Rasterizer_mesh(opt=opt, mesh_path=mesh_path)
        elif opt.pbr:
            model = Rasterizer_pbr(opt=opt, tet_res=opt.tet_res, mesh_path=mesh_path)
        else:
            model = Rasterizer_MLP(opt=opt, tet_res=opt.tet_res, mesh_path=mesh_path)

        resolution = opt.res_fine
        size = 800 * nprocs
        train_loader_s0 = TrainDataset(opt, device=device, type='train',
                                       H=resolution, W=resolution, size=size, focal=0,
                                       targets=None, weights=None, radius=None).dataloader(opt.world_size, opt.bs)
        valid_loader = RastDataset(opt, device=device, type='val',
                                   H=resolution, W=resolution, size=1).dataloader(opt.world_size, opt.bs)

        if opt.optimizer == 'adam':
            optimizer = lambda params: torch.optim.Adam(params, betas=(0.9, 0.99), eps=1e-15)
        else:
            optimizer = lambda params: torch.optim.AdamW(params, betas=(0.9, 0.99), eps=1e-15)

        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1.0)

        trainer = FineTrainer('df', opt, model, guidance, guidance_others, device=device,
                              workspace=opt.workspace, optimizer=optimizer, lr_scheduler=scheduler,
                              use_checkpoint=opt.ckpt_fine, eval_interval=opt.eval_interval,
                              local_rank=opt.local_rank, world_size=opt.world_size)

        # init sdf mlp
        if not opt.fix_mesh and opt.ckpt_fine == 'scratch':
            query_points = model.points
            sdf = model.sdf
            texture = None if (not opt.init_texture) or opt.latent else model.texture
            trainer.train_initsdf(query_points, sdf, texture)

        if opt.fix_mesh and opt.init_texture:
            points, texture = model.init_texture()
            if texture is not None:
                trainer.train_initsdf_fixmesh(points, texture)

        max_epoch = np.ceil(opt.iters / len(train_loader_s0)).astype(np.int32)
        if not opt.single_stage:
            if max_epoch <= 3:
                max_epoch_s0 = max_epoch
                max_epoch_s1 = 0
                max_epoch_s2 = 0
            elif max_epoch <= 9:
                max_epoch_s0 = 2
                max_epoch_s1 = 2
                max_epoch_s2 = max_epoch - 4
            else:
                max_epoch_s0 = np.ceil(0.3 * max_epoch).astype(np.int32)
                max_epoch_s1 = np.ceil(0.3 * max_epoch).astype(np.int32)
                max_epoch_s2 = np.ceil(0.4 * max_epoch).astype(np.int32)
        else:
            max_epoch_s0 = max_epoch
            max_epoch_s1 = 0
            max_epoch_s2 = 0
        print(f'max epoch: {max_epoch_s0}+{max_epoch_s1}+{max_epoch_s2}')

        # s0
        if opt.ls_all:
            trainer.module.compute_targets()
            targets = trainer.module.targets
            weights = trainer.module.weights
            radius = trainer.module.radius
        else:
            targets = None
            weights = None
            radius = None
        train_loader_s0 = TrainDataset(
            opt, device=device, type='train',
            H=resolution, W=resolution, size=size, focal=0,
            targets=targets, weights=weights, radius=radius
        ).dataloader(opt.world_size, opt.bs)
        trainer.train(train_loader_s0, valid_loader, max_epoch_s0, 0)

        # s1
        trainer.module.compute_targets()
        targets = trainer.module.targets
        weights = trainer.module.weights
        radius = trainer.module.radius
        train_loader_s1 = TrainDataset(
            opt, device=device, type='train',
            H=resolution, W=resolution, size=size, focal=1,
            targets=targets, weights=weights, radius=radius
        ).dataloader(opt.world_size, opt.bs)
        trainer.train(train_loader_s1, valid_loader, max_epoch_s0 + max_epoch_s1, 0)

        # s2
        trainer.module.compute_targets()
        targets = trainer.module.targets
        weights = trainer.module.weights
        radius = trainer.module.radius
        train_loader_s2 = TrainDataset(
            opt, device=device, type='train',
            H=resolution, W=resolution, size=size, focal=2,
            targets=targets, weights=weights, radius=radius
        ).dataloader(opt.world_size, opt.bs)
        trainer.train(train_loader_s2, valid_loader, max_epoch_s0 + max_epoch_s1 + max_epoch_s2, 0)

        test_loader = RastDataset(opt, device=device, type='test',
                                  H=resolution, W=resolution, size=100).dataloader()
        trainer.test(test_loader)

        if opt.save_mesh and not opt.latent:
            trainer.save_mesh()


if __name__ == '__main__':
    opt = config_parser()
    opt.workspace = os.path.join('exp', opt.workspace)

    if opt.O0:  # pixel-wise grad suppress
        opt.fp16 = True
        opt.grad_clip_rgb = 0.1
        opt.grad_suppress_type = 0

    if opt.O1:  # text-to-3d
        opt.fp16 = True
        opt.dir_text = True
        opt.save_mesh = True
        opt.canonicalize = True
        opt.init_texture = True
    elif opt.O2:  # text-to-texture
        opt.fp16 = True
        opt.dir_text = True
        opt.save_mesh = True
        opt.canonicalize = True
        opt.fix_mesh = True
        opt.no_normal = True
        # opt.dreamtime = True
        if opt.O4:
            opt.guidance += "xl"
        else:
            opt.guidance = "controlnet"
    elif opt.O3:  # image-to-3d
        opt.fp16 = True
        opt.dir_text = True
        opt.save_mesh = True
        # 0123
        opt.zero123 = True
        opt.bgt = True
        opt.no_normal = True
        opt.radius_range = [1.0, 1.5]
        opt.n_targets = 0
        opt.smooth_iters = opt.iters
        opt.lambda_smooth = opt.lambda_smooth / 2
        opt.lr_fine = [1e-6, 1e-3]  # small shape lr
    elif opt.O4:  # text-to-3d using SDXL
        opt.fp16 = True
        opt.dir_text = True
        opt.save_mesh = True
        opt.canonicalize = True
        # SDXL
        opt.guidance += "xl"

    if opt.fix_mesh:
        opt.no_normal = True

    if 'xl' in opt.guidance:
        opt.bg_scenery = True
        # opt.no_normal = True  # vram is not big enough
        if (not opt.no_normal and not opt.only_normal) or opt.vsd:
            opt.bs = 1
            opt.iters = 2400
        else:
            opt.bs = 2
            opt.iters = 1200
        opt.res_fine = 1024
        opt.optimizer = "adam"
        opt.scheduler = "dpm"
        opt.smooth_iters = 0

    if opt.ref_path:
        opt.focal_range = [1.025, 1.025]
        opt.focal_range_fine = [1.025, 1.025]

    if opt.no_normal:
        opt.lr_fine = [opt.lr_fine[0] / 20, opt.lr_fine[1]]

    if opt.latent:
        opt.albedo = True
        opt.res_fine = 256 if 'xl' in opt.guidance else 128

    if opt.albedo:
        opt.albedo_iters = opt.iters

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    opt.nprocs = len(opt.gpus.split(','))

    mp.spawn(main_worker, nprocs=opt.nprocs, args=(opt.nprocs, opt))
