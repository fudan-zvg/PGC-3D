import os
import cv2
import glob
import json
import tqdm
import random
import numpy as np

import trimesh

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils import get_rays, safe_normalize

DIR_COLORS = np.array([
    [255, 0, 0, 255],  # front
    [0, 255, 0, 255],  # side
    [0, 0, 255, 255],  # back
    [255, 255, 0, 255],  # side
    [255, 0, 255, 255],  # overhead
    # [0, 255, 255, 255], # bottom
], dtype=np.uint8)


def cartesian_to_spherical(xyz):
    xy = xyz[..., 0] ** 2 + xyz[..., 1] ** 2
    z = torch.sqrt(xy + xyz[:, 2] ** 2)
    theta = torch.arctan2(torch.sqrt(xy), xyz[..., 2])  # for elevation angle defined from Z-axis down
    # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    azimuth = torch.arctan2(xyz[..., 1], xyz[..., 0])
    return torch.stack([theta, azimuth, z], dim=-1)


def get_T(T_target, T_cond):
    spherical_cond = cartesian_to_spherical(T_cond)
    spherical_target = cartesian_to_spherical(T_target)

    d_theta = spherical_target[..., 0] - spherical_cond[..., 0]
    d_azimuth = (spherical_target[..., 1] - spherical_cond[..., 1]) % (2 * np.pi)
    d_z = spherical_target[..., 2] - spherical_cond[..., 2]

    d_T = torch.stack([d_theta, torch.sin(d_azimuth),
                       torch.cos(d_azimuth), d_z], dim=-1)
    return d_T


def visualize_poses(poses, dirs, size=0.1):
    # poses: [B, 4, 4], dirs: [B]

    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=1)
    objects = [axes, sphere]

    for pose, dir in zip(poses, dirs):
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])
        segs = trimesh.load_path(segs)

        # different color for different dirs
        segs.colors = DIR_COLORS[[dir]].repeat(len(segs.entities), 0)

        objects.append(segs)

    trimesh.Scene(objects).show()


def get_view_direction(thetas, phis, overhead, front, direction='right'):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (left) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (right) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    if direction == 'left':
        dirs = [2, 3, 0, 1, 4, 5]
    else:
        dirs = [0, 1, 2, 3, 4, 5]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis
    res[(phis < front)] = dirs[0]
    res[(phis >= front) & (phis < np.pi)] = dirs[1]
    res[(phis >= np.pi) & (phis < (np.pi + front))] = dirs[2]
    res[(phis >= (np.pi + front))] = dirs[3]
    # override by thetas
    res[thetas <= overhead] = dirs[4]
    res[thetas >= (np.pi - overhead)] = dirs[5]
    return res


def rand_poses(size, device, radius_range=[1, 1.5], theta_range=[45, 100], phi_range=[0, 360], return_dirs=False,
               angle_overhead=30, angle_front=60, jitter=False, uniform_sphere_rate=0.5,
               target_center=None, target_radius=None, direction='right'):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''

    theta_range = np.deg2rad(theta_range)
    phi_range = np.deg2rad(phi_range)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)

    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]

    if random.random() < uniform_sphere_rate:
        unit_centers = F.normalize(
            torch.stack([
                (torch.rand(size, device=device) - 0.5) * 2.0,
                torch.rand(size, device=device),
                (torch.rand(size, device=device) - 0.5) * 2.0,
            ], dim=-1), p=2, dim=1
        )
        thetas = torch.acos(unit_centers[:, 1])
        phis = torch.atan2(unit_centers[:, 0], unit_centers[:, 2])
        phis[phis < 0] += 2 * np.pi
        centers = unit_centers * radius.unsqueeze(-1)
    else:
        thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
        phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

        centers = torch.stack([
            torch.sin(thetas) * torch.sin(phis) * radius,
            torch.cos(thetas) * radius,
            torch.sin(thetas) * torch.cos(phis) * radius,
        ], dim=-1)  # [B, 3]

    targets = 0

    # jitters
    if jitter:
        centers = centers + (torch.rand_like(centers) * 0.2 - 0.1)
        targets = targets + torch.randn_like(centers) * 0.447

    # lookat
    forward_vector = safe_normalize(targets - centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    if jitter:
        up_noise = torch.randn_like(up_vector) * 0.141
        up_vector = safe_normalize(up_vector + up_noise)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))

    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    if target_center is not None:
        target_center = torch.tensor(target_center, dtype=centers.dtype, device=device).reshape(1, 3)
        centers[:] = centers[:] * max(min(target_radius, 0.5), 0.3) + target_center
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front, direction)
    else:
        dirs = None

    return poses, dirs, thetas, phis, radius


def circle_poses(device, radius=1.25, theta=60, phi=0, return_dirs=False, angle_overhead=30, angle_front=60):
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)

    thetas = torch.FloatTensor([theta]).to(device)
    phis = torch.FloatTensor([phi]).to(device)

    centers = torch.stack([
        torch.sin(thetas) * torch.sin(phis) * radius,
        torch.cos(thetas) * radius,
        torch.sin(thetas) * torch.cos(phis) * radius,
    ], dim=-1)  # [B, 3]

    # lookat
    forward_vector = - safe_normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = None

    return poses, dirs


class RastDataset:
    def __init__(self, opt, device, type='train', H=512, W=512, size=100, focal=0,
                 targets=None, weights=None, radius=None):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type  # train, val, test

        self.H = H
        self.W = W
        self.radius_range = opt.radius_range
        # self.fovy_range = opt.fovy_range
        if opt.test or focal == 0:
            self.focal_range = opt.focal_range
        elif focal == 1:
            self.focal_range = [(opt.focal_range_fine[0] + opt.focal_range[0]) / 2,
                                (opt.focal_range_fine[1] + opt.focal_range[1]) / 2]
        elif focal == 2:
            self.focal_range = opt.focal_range_fine
        self.size = size

        self.training = self.type in ['train', 'all']

        self.cx = self.H / 2
        self.cy = self.W / 2

        self.targets = targets
        self.weights = weights
        self.radius = radius

        # [debug] visualize poses
        # poses, dirs = rand_poses(100, self.device, radius_range=self.radius_range, return_dirs=self.opt.dir_text, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front, jitter=self.opt.jitter_pose, uniform_sphere_rate=1)
        # visualize_poses(poses.detach().cpu().numpy(), dirs.detach().cpu().numpy())

    def collate(self, index):
        B = len(index)  # always 1
        # print("collate", index, B)

        if self.training:
            # random pose on the fly
            if self.targets is None:
                target_center = None
                target_radius = None
                part = 0
            else:
                k = self.targets.shape[0]
                if self.weights is None:
                    self.weights = np.ones(k) / k
                prob = np.cumsum(self.weights) * self.opt.local_prob

                rand = random.random()
                target_center = None
                target_radius = None
                part = 0
                for i in range(k):
                    if rand <= prob[k - i - 1]:
                        target_center = self.targets[k - i - 1]
                        target_radius = self.radius[k - i - 1]
                        part = k - i

            direction = 'left' if self.opt.left else 'right'
            poses, dirs, theta, phi, radius = rand_poses(B, self.device, radius_range=self.radius_range,
                                                         return_dirs=self.opt.dir_text,
                                                         angle_overhead=self.opt.angle_overhead,
                                                         angle_front=self.opt.angle_front,
                                                         jitter=self.opt.jitter_pose,
                                                         uniform_sphere_rate=self.opt.uniform_sphere_rate,
                                                         target_center=target_center,
                                                         target_radius=target_radius,
                                                         direction=direction
                                                         )

            # random focal
            # fov = random.random() * (self.fovy_range[1] - self.fovy_range[0]) + self.fovy_range[0]
            # focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
            focal = self.H * (random.random() * (self.focal_range[1] - self.focal_range[0]) + self.focal_range[0])
            intrinsics = np.array([focal, focal, self.cx, self.cy])
        else:
            part = 0
            # circle pose
            theta = 60
            phi = (index[0] / self.size) * 360
            # phi = 135
            radius = (self.radius_range[1] + self.radius_range[0]) / 2
            poses, dirs = circle_poses(self.device,
                                       radius=radius,
                                       theta=theta, phi=phi,
                                       return_dirs=self.opt.dir_text,
                                       angle_overhead=self.opt.angle_overhead,
                                       angle_front=self.opt.angle_front)

            # fixed focal
            # fov = (self.fovy_range[1] + self.fovy_range[0]) / 2
            # focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
            focal = self.H * (self.focal_range[1] + self.focal_range[0]) / 2
            intrinsics = np.array([focal, focal, self.cx, self.cy])

        # sample a low-resolution but full image for CLIP
        rays = get_rays(poses, intrinsics, self.H, self.W, -1)

        # reference
        dT = None
        if self.opt.ref_path and self.training:
            # phi = 0, 90, 180, 270
            rr = 2.5 / 1.8
            pose_0 = torch.tensor([[[-1.0000, 0.0000, -0.0000, 0.0000 * rr],
                                    [0.0000, -0.8660, -0.5000, 0.9000 * rr],
                                    [0.0000, 0.5000, -0.8660, 1.5588 * rr],
                                    [0.0000, 0.0000, 0.0000, 1.0000]]])
            pose_90 = torch.tensor([[[4.3711e-08, 5.0000e-01, -8.6603e-01, 1.5588e+00],
                                     [0.0000e+00, -8.6603e-01, -5.0000e-01, 9.0000e-01],
                                     [1.0000e+00, -2.1856e-08, 3.7855e-08, -6.8139e-08],
                                     [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]]])
            pose_180 = torch.tensor([[[1.0000e+00, -4.3711e-08, 7.5710e-08, -1.3628e-07],
                                      [0.0000e+00, -8.6603e-01, -5.0000e-01, 9.0000e-01],
                                      [-8.7423e-08, -5.0000e-01, 8.6603e-01, -1.5588e+00],
                                      [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]]])
            pose_270 = torch.tensor([[[-1.1925e-08, -5.0000e-01, 8.6603e-01, -1.5588e+00],
                                      [-0.0000e+00, -8.6603e-01, -5.0000e-01, 9.0000e-01],
                                      [-1.0000e+00, 5.9624e-09, -1.0327e-08, 1.8589e-08],
                                      [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]]])
            pose_12 = torch.tensor([[[-0.7289686, 0.34227353, -0.59283525, 1.1856705],
                                     [0., -0.8660255, -0.49999997, 0.99999994],
                                     [0.6845471, 0.36448428, -0.6313054, 1.2626108],
                                     [0., 0., 0., 1.]]])
            # right
            pose_right = torch.tensor([[[7.0711e-01, -3.0909e-08, -7.0711e-01, 8.8750e-01],
                                        [0.0000e+00, -1.0000e+00, 4.3711e-08, -5.4863e-08],
                                        [7.0711e-01, 3.0909e-08, 7.0711e-01, -8.8750e-01],
                                        [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]]])
            pose_left = torch.tensor([[[-7.0711e-01, 3.0909e-08, 7.0711e-01, -8.8750e-01],
                                       [-0.0000e+00, -1.0000e+00, 4.3711e-08, -5.4863e-08],
                                       [-7.0711e-01, -3.0909e-08, -7.0711e-01, 8.8750e-01],
                                       [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]]])
            if self.opt.left:
                pose_ref = torch.cat([pose_left], dim=0).to(poses.device)
            else:
                pose_ref = torch.cat([pose_right], dim=0).to(poses.device)

            pose_ref = torch.cat([pose_0], dim=0).to(poses.device)

            # get dT
            convert = lambda x: x[..., [2, 0, 1]]
            dT = get_T(T_target=convert(poses[:, :3, 3]),
                       T_cond=convert(pose_ref[0:1, :3, 3])).to(self.device)
            poses = torch.cat([poses, pose_ref], dim=0)

        data = {
            'H': self.H,
            'W': self.W,
            'focal': focal,
            'poses': poses,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'dir': dirs,
            'part': part,
            'index': torch.tensor(index, device=self.device),
            'dT': dT
        }

        return data

    def dataloader(self, world_size=1, per_gpu_batch_size=1):
        dataset = list(range(self.size))

        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        loader = DataLoader(dataset, batch_size=per_gpu_batch_size,
                            collate_fn=self.collate,
                            num_workers=0, sampler=sampler)
        return loader


class RecDataset:
    def __init__(self, opt, device, type='train', H=512, W=512, **kwargs):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type  # train, val, test

        self.H = H
        self.W = W
        self.cx = self.H / 2
        self.cy = self.W / 2
        self.focal = self.H * (self.opt.focal_range[1] + self.opt.focal_range[0]) / 2
        self.intrinsics = np.array([self.focal, self.focal, self.cx, self.cy])

        # from opt.data_root load images and poses
        rgb_path = os.path.join(self.opt.data_root, 'rgb')
        files = os.listdir(rgb_path)
        files.sort()
        files = [os.path.join(rgb_path, file) for file in files]

        imgs = []
        for file in files:
            if file.endswith('.png'):
                img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
                img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB) / 255
                img = torch.tensor(img, dtype=torch.float32, device=self.device)[None, ...]
                imgs.append(img)

        poses = np.load(os.path.join(self.opt.data_root, 'poses.npy'))
        self.poses = torch.from_numpy(poses).to(self.device)
        self.size = poses.shape[0]

        images = torch.cat(imgs, dim=0).permute(0, 3, 1, 2)
        res = self.opt.res_fine
        self.images = F.interpolate(images, (res, res), mode='bilinear', align_corners=False)

        self.training = self.type in ['train', 'all']

    def collate(self, index):
        B = len(index)
        # print("collate", index, B)

        poses = self.poses[index]
        images = self.images[index]
        rays = get_rays(poses, self.intrinsics, self.H, self.W, -1)

        # dirs
        unit_centers = F.normalize(
            poses[:, :3, 3], p=2, dim=1
        )
        thetas = torch.acos(unit_centers[:, 1])
        phis = torch.atan2(unit_centers[:, 0], unit_centers[:, 2])
        phis[phis < 0] += 2 * np.pi

        angle_overhead = np.deg2rad(self.opt.angle_overhead)
        angle_front = np.deg2rad(self.opt.angle_front)
        direction = 'left' if self.opt.left else 'right'
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front, direction)

        data = {
            'H': self.H,
            'W': self.W,
            'focal': self.focal,
            'poses': poses,
            'images': images,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'dir': dirs,
            'part': 0,
            'index': torch.tensor(index, device=self.device),
            'dT': None
        }

        return data

    def dataloader(self, world_size=1, per_gpu_batch_size=1):
        dataset = list(range(self.size))

        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        loader = DataLoader(dataset, batch_size=per_gpu_batch_size,
                            collate_fn=self.collate,
                            num_workers=0, sampler=sampler)
        return loader
