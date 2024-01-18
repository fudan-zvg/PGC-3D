import os
import nvdiffrast.torch as dr
import torch
import cv2

import open3d as o3d
from nerf.fine.perspective_camera import PerspectiveCamera
from nerf.fine.rast_utils import *
from nerf.fine.neural_render import NeuralRender


class Rasterizer_mesh(nn.Module):
    def __init__(
            self,
            opt,
            device='cuda',
            mesh_path=None
    ):
        super().__init__()
        self.opt = opt
        self.device = device
        self.mesh_path = mesh_path
        self.threshold = 99  # remove floaters
        self.ctx = dr.RasterizeCudaContext(device=self.device)

        # for sampling
        self.targets = None
        self.weights = None
        self.radius = None

        # texture
        if self.opt.pbr:
            kd_min = [0.03, 0.03, 0.03, 0.03]  # Limits for kd
            kd_max = [0.97, 0.97, 0.97, 0.97]
            ks_min = opt.ks_min  # Limits for ks
            ks_max = opt.ks_max
            nrm_min = [-1.0, -1.0, 0.0]  # Limits for normal map
            nrm_max = [1.0, 1.0, 1.0]

            kd_min, kd_max = torch.tensor(kd_min, dtype=torch.float, device=device), \
                             torch.tensor(kd_max, dtype=torch.float, device=device)
            ks_min, ks_max = torch.tensor(ks_min, dtype=torch.float, device=device), \
                             torch.tensor(ks_max, dtype=torch.float, device=device)
            nrm_min, nrm_max = torch.tensor(nrm_min, dtype=torch.float, device=device), \
                               torch.tensor(nrm_max, dtype=torch.float, device=device)
            mlp_min = torch.cat((kd_min[0:3], ks_min, nrm_min), dim=0)
            mlp_max = torch.cat((kd_max[0:3], ks_max, nrm_max), dim=0)
            mlp_map_opt = Material_mlp(min_max=[mlp_min, mlp_max])
            self.mat = material.Material({'kd_ks_normal': mlp_map_opt,
                                          'no_perturbed_nrm': opt.no_perturbed_nrm})
            self.mat['bsdf'] = 'pbr'
            self.kd_min_max = [kd_min, kd_max]
            self.ks_min_max = [ks_min, ks_max]
            self.nrm_min_max = [nrm_min, nrm_max]

            # env
            # self.lgt = light.create_trainable_env_rnd(512, scale=0, bias=0.75)
            # envmap = "others/nvdiffrec/data/irrmaps/aerodynamics_workshop_2k.hdr"
            envmap = "others/nvdiffrec/data/irrmaps/mud_road_puresky_4k.hdr"
            self.lgt = light.load_env(envmap, scale=2.0)
            with torch.no_grad():
                self.lgt.build_mips()
        else:
            self.renderer = NeuralRender(device)
            self.encoder_tex, self.in_dim = get_encoder('hashgrid', input_dim=3,
                                                        log2_hashmap_size=16,
                                                        desired_resolution=2 ** 12,
                                                        base_resolution=2 ** 4,
                                                        level_dim=4)
            out_dim = 4 if self.opt.latent else 3
            self.texture_MLP = MLP(self.in_dim, out_dim, 32, 2, bias=True)

        if self.opt.bg_scenery:
            img = cv2.imread("data/background.png", cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB) / 255
            precision_t = torch.float32  # only support float32
            img = torch.tensor(img, dtype=precision_t, device=self.device)
            self.bg = img[None, ...]

        # init geometry
        self.init_geometry(canonicalize=opt.canonicalize)

    def get_texture(self, x):
        if self.opt.pbr:
            texture = self.mat['kd_ks_normal'].sample(x)
        else:
            hidden = self.encoder_tex(x, bound=self.opt.bound)
            texture = self.texture_MLP(hidden)
            if not self.opt.latent:
                texture = torch.sigmoid(texture) * 0.9 + 0.05  # [0.05, 0.95]

        return texture

    def export_mesh(self, path, **kwargs):

        if self.opt.pbr:
            base_mesh = xatlas_uvmap(self.ctx, self.imesh, self.mat, self.device,
                                     self.kd_min_max, self.ks_min_max, self.nrm_min_max)
            obj.write_obj(path, base_mesh)
            light.save_env_map(os.path.join(path, "probe.hdr"), self.lgt)
        else:
            def _export(v, f, h0=2048, w0=2048, ssaa=1, name=''):
                # v, f: torch Tensor
                device = v.device
                v_np = v.detach().cpu().numpy()  # [N, 3]
                f_np = f.detach().cpu().numpy()  # [M, 3]

                print(f'[INFO] running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}')

                # unwrap uvs
                import xatlas
                import nvdiffrast.torch as dr
                from sklearn.neighbors import NearestNeighbors
                from scipy.ndimage import binary_dilation, binary_erosion

                glctx = dr.RasterizeCudaContext()

                atlas = xatlas.Atlas()
                atlas.add_mesh(v_np, f_np)
                chart_options = xatlas.ChartOptions()
                chart_options.max_iterations = 0  # disable merge_chart for faster unwrap...
                atlas.generate(chart_options=chart_options)
                vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

                # vmapping, ft_np, vt_np = xatlas.parametrize(v_np, f_np) # [N], [M, 3], [N, 2]

                vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
                ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)

                # render uv maps
                uv = vt * 2.0 - 1.0  # uvs to range [-1, 1]
                uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1)  # [N, 4]

                if ssaa > 1:
                    h = int(h0 * ssaa)
                    w = int(w0 * ssaa)
                else:
                    h, w = h0, w0

                rast, _ = dr.rasterize(glctx, uv.unsqueeze(0), ft, (h, w))  # [1, h, w, 4]
                xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f)  # [1, h, w, 3]
                mask, _ = dr.interpolate(torch.ones_like(v[:, :1]).unsqueeze(0), rast, f)  # [1, h, w, 1]

                # masked query
                xyzs = xyzs.view(-1, 3)
                mask = (mask > 0).view(-1)
                feats = torch.zeros(h * w, 3, device=device, dtype=torch.float32)

                if mask.any():
                    xyzs = xyzs[mask]  # [M, 3]

                    # batched inference to avoid OOM
                    all_feats = []
                    head = 0
                    while head < xyzs.shape[0]:
                        tail = min(head + 640000, xyzs.shape[0])
                        texture = self.get_texture(xyzs[head:tail])
                        all_feats.append(texture)
                        head += 640000

                    feats[mask] = torch.cat(all_feats, dim=0)

                feats = feats.view(h, w, -1)
                mask = mask.view(h, w)

                # quantize [0.0, 1.0] to [0, 255]
                feats = feats.detach().cpu().numpy()
                feats = (feats * 255).astype(np.uint8)

                ### NN search as an antialiasing ...
                mask = mask.cpu().numpy()

                inpaint_region = binary_dilation(mask, iterations=3)
                inpaint_region[mask] = 0

                search_region = mask.copy()
                not_search_region = binary_erosion(search_region, iterations=2)
                search_region[not_search_region] = 0

                search_coords = np.stack(np.nonzero(search_region), axis=-1)
                inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

                knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_coords)
                _, indices = knn.kneighbors(inpaint_coords)

                feats[tuple(inpaint_coords.T)] = feats[tuple(search_coords[indices[:, 0]].T)]

                # do ssaa after the NN search, in numpy
                feats = cv2.cvtColor(feats, cv2.COLOR_RGB2BGR)

                if ssaa > 1:
                    # alphas = cv2.resize(alphas, (w0, h0), interpolation=cv2.INTER_NEAREST)
                    feats = cv2.resize(feats, (w0, h0), interpolation=cv2.INTER_LINEAR)

                cv2.imwrite(os.path.join(path, f'{name}fine_albedo.png'), feats)

                # save obj (v, vt, f /)
                obj_file = os.path.join(path, f'{name}fine_mesh.obj')
                mtl_file = os.path.join(path, f'{name}fine_mesh.mtl')

                print(f'[INFO] writing obj mesh to {obj_file}')
                with open(obj_file, "w") as fp:
                    fp.write(f'mtllib {name}fine_mesh.mtl \n')

                    print(f'[INFO] writing vertices {v_np.shape}')
                    for v in v_np:
                        fp.write(f'v {-v[0]} {v[1]} {v[2]} \n')

                    print(f'[INFO] writing vertices texture coords {vt_np.shape}')
                    for v in vt_np:
                        fp.write(f'vt {v[0]} {1 - v[1]} \n')

                    print(f'[INFO] writing faces {f_np.shape}')
                    fp.write(f'usemtl mat0 \n')
                    for i in range(len(f_np)):
                        fp.write(
                            f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} {f_np[i, 1] + 1}/{ft_np[i, 1] + 1} {f_np[i, 2] + 1}/{ft_np[i, 2] + 1} \n")

                with open(mtl_file, "w") as fp:
                    fp.write(f'newmtl mat0 \n')
                    fp.write(f'Ka 1.000000 1.000000 1.000000 \n')
                    fp.write(f'Kd 1.000000 1.000000 1.000000 \n')
                    fp.write(f'Ks 0.000000 0.000000 0.000000 \n')
                    fp.write(f'Tr 1.000000 \n')
                    fp.write(f'illum 1 \n')
                    fp.write(f'Ns 0.000000 \n')
                    fp.write(f'map_Kd {name}fine_albedo.png \n')

            faces = torch.cat(
                [self.f[:, 0:1],
                 self.f[:, 2:3],
                 self.f[:, 1:2], ], dim=-1)
            _export(self.v, faces.int())
        return

    def render_mesh(self, mesh_v_nx3, mesh_f_fx3, mesh_vn_nx3, textures_nx3, camera_mv_bx4x4,
                    resolution=512, hierarchical_mask=False):
        return_value = dict()

        B = camera_mv_bx4x4.shape[0]
        mesh_v_bxnx3 = mesh_v_nx3.unsqueeze(dim=0).expand([B, -1, 3])
        textures_bxnx3 = textures_nx3.unsqueeze(dim=0).expand([B, -1, 3])
        mesh_vn_bxnx3 = mesh_vn_nx3.unsqueeze(dim=0).expand([B, -1, 3])
        mesh_feat_bxnx6 = torch.cat([mesh_v_bxnx3, textures_bxnx3, mesh_vn_bxnx3], dim=-1)
        feats, mask, hard_mask, rast, v_pos_clip, mask_pyramid, depth, antialias_albedo = self.renderer.render_mesh(
            mesh_v_bxnx3,
            mesh_f_fx3.int(),
            camera_mv_bx4x4,
            mesh_feat_bxnx6,
            resolution=resolution,
            device=self.device,
            hierarchical_mask=hierarchical_mask
        )

        return_value['tex_pos'] = feats[..., :3]
        return_value['albedo'] = feats[..., 3:6]
        return_value['normals'] = feats[..., 6:9]
        return_value['antialias_albedo'] = antialias_albedo
        return_value['mask'] = mask
        return_value['hard_mask'] = hard_mask
        return_value['rast'] = rast
        return_value['v_pos_clip'] = v_pos_clip
        return_value['mask_pyramid'] = mask_pyramid
        return_value['depth'] = depth

        return return_value

    def forward(self, data, eval=False, **kwargs):
        focal = data['focal']
        cam_mv = data['poses']  # Bx4x4
        H, W = data['H'], data['W']
        B = cam_mv.shape[0]

        focal = H / (2 * focal)
        camera = PerspectiveCamera(focal=focal, device=self.device)

        # Generate 3D mesh first
        if self.opt.bg_scenery and not eval:
            bg_color = F.interpolate(self.bg.permute(0, 3, 1, 2),
                                     (H, W), mode='bilinear', align_corners=False)
            background = bg_color.permute(0, 2, 3, 1).expand(B, H, W, 3)
        else:
            background = torch.ones((B, H, W, 3), dtype=torch.float32, device=self.device) * self.opt.bg_color

        if self.opt.latent:
            background = 0.0

        if self.opt.pbr:
            proj_mtx = camera.proj_mtx
            mv = torch.linalg.inv(cam_mv)
            mvp = (proj_mtx @ mv).float()
            campos = cam_mv[:, :3, 3]
            # self.lgt.build_mips()
            # self.lgt.xfm(mv)
            buffers = render.render_mesh(self.ctx, self.imesh, mvp, mv, campos, self.lgt, [H, W],
                                         spp=1, msaa=True, background=background, bsdf=None)

            # Render the normal into 2D image
            img_antilias = buffers['shaded'][..., 0:3]

            buffers['normals'] = -buffers['normals']

            normals_vis = (-buffers['normals'] + 1) / 2
            mask = buffers['mask']
            buffers['normals_vis'] = normals_vis * mask + (1 - mask)

            # Albedo (k_d) smoothnesss regularizer
            buffers['reg_kd_smooth'] = torch.mean(buffers['kd_grad'][..., :-1] * buffers['kd_grad'][..., -1:])
            buffers['reg_ks_smooth'] = torch.mean(buffers['ks_grad'][..., :-1] * buffers['ks_grad'][..., -1:])

            # Visibility regularizer
            # buffers['reg_vis'] = torch.mean(buffers['occlusion'][..., :-1] * buffers['occlusion'][..., -1:])

            # Light white balance regularizer
            # buffers['reg_lgt'] = self.lgt.regularizer()
        else:
            self.renderer.camera = camera
            buffers = self.render_mesh(self.v, self.f, self.vn, self.v, cam_mv, resolution=H)
            tex_pos = buffers['tex_pos']
            mask = buffers['mask']
            buffers['normals'] = -buffers['normals']
            normals_vis = (-buffers['normals'] + 1) / 2
            buffers['normals_vis'] = normals_vis * mask + (1 - mask)

            if not eval and self.opt.pos_perturb:
                tex_pos = tex_pos + (torch.rand(tex_pos.size(), device=self.device) * 2e-3)

            # texture
            albedo = self.get_texture(tex_pos)
            img_antilias = albedo * mask + background * (1 - mask)

        return img_antilias, buffers

    def get_params(self, lr):
        if self.opt.pbr:
            params = [
                {'params': self.mat.parameters(), 'lr': lr[1]},
                {'params': self.lgt.parameters(), 'lr': lr[1]}
            ]
        else:
            params = [
                {'params': self.encoder_tex.parameters(), 'lr': lr[1] * 10},
                {'params': self.texture_MLP.parameters(), 'lr': lr[1]},
            ]
        return params

    def compute_targets(self):
        if self.opt.n_targets > 0:
            v = self.imesh.v_pos if self.opt.pbr else self.v
            vn = self.imesh.v_nrm if self.opt.pbr else self.vn
            vn = vn.detach().cpu().numpy()
            vc = self.get_texture(v)[..., :3].detach().cpu().numpy()
            targets, weights, radius, _ = compute_targets(v.detach().cpu().numpy(), vn, self.opt.n_targets, vc)
            self.targets = targets
            self.weights = weights
            self.radius = radius
            print(f'targets: {self.targets}')
            print(f'weights: {self.weights}')
            print(f'radius: {self.radius}')
        return

    def init_geometry(self, canonicalize=True, remove_floaters=False):
        mesh0 = o3d.io.read_triangle_mesh(self.mesh_path)
        if remove_floaters:
            print("removing floaters...")
            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                triangle_clusters, cluster_n_triangles, cluster_area = mesh0.cluster_connected_triangles()
            triangle_clusters = np.asarray(triangle_clusters)
            cluster_n_triangles = np.asarray(cluster_n_triangles)

            threshold = np.percentile(cluster_n_triangles, self.threshold)
            triangles_to_remove = cluster_n_triangles[triangle_clusters] < threshold
            mesh0.remove_triangles_by_mask(triangles_to_remove)

        v = np.asarray(mesh0.vertices)
        f = np.asarray(mesh0.triangles)
        # canonicalize mesh
        if canonicalize:
            vcenter = (np.max(v, 0) + np.min(v, 0)) / 2
            v = v - vcenter
            scale = 0.8 / np.abs(v).max()
            v = v * scale

        verts = torch.tensor(v, dtype=torch.float32, device=self.device)
        faces = torch.tensor(f, dtype=torch.long, device=self.device)

        if self.opt.pbr:
            atlas = xatlas.Atlas()
            atlas.add_mesh(v, f.astype(np.int64))
            chart_options = xatlas.ChartOptions()
            chart_options.max_iterations = 0  # disable merge_chart for faster unwrap...
            atlas.generate(chart_options=chart_options)
            vmapping, indices, uvs = atlas[0]  # [N], [M, 3], [N, 2]

            # vmapping, indices, uvs = xatlas.parametrize(v, f)

            # Convert to tensors
            indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)
            uvs = torch.tensor(uvs, dtype=torch.float, device=self.device)
            uv_idx = torch.tensor(indices_int64, dtype=torch.int64, device=self.device)

            from others.nvdiffrec.render import mesh
            imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=self.mat)

            # Run mesh operations to generate tangent space
            imesh = mesh.auto_normals(imesh)
            self.imesh = mesh.compute_tangents(imesh)
        else:
            self.v = verts
            self.f = faces

            from pytorch3d.structures import Meshes
            mesh = Meshes(verts=[verts], faces=[faces])
            mesh_vn = mesh.verts_normals_packed()
            self.vn = mesh_vn

        return

    def init_texture(self, canonicalize=True):
        mesh0 = o3d.io.read_triangle_mesh(self.mesh_path)
        v = np.asarray(mesh0.vertices)
        f = np.asarray(mesh0.triangles)
        vc = np.asarray(mesh0.vertex_colors)

        if canonicalize:
            vcenter = (np.max(v, 0) + np.min(v, 0)) / 2
            v = v - vcenter
            scale = 0.8 / np.abs(v).max()
            v = v * scale

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(v)
        mesh.triangles = o3d.utility.Vector3iVector(f)
        mesh.vertex_colors = o3d.utility.Vector3dVector(vc)

        # get points
        n_points = 1000000 // self.opt.nprocs
        pcd = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, number_of_points=n_points)
        query_points = np.asarray(pcd.points)
        query_points = query_points + np.random.normal(0, 1, n_points * 3).reshape([-1, 3]) * 0.04
        # points away from surface (avoid floaters)
        query_points_out = query_points + np.random.normal(0, 1, n_points * 3).reshape([-1, 3]) * 0.4
        query_points = np.concatenate((query_points, query_points_out), axis=0)
        query_points = query_points.astype(np.float32)

        # texture
        texture = None
        if mesh0.has_vertex_colors():
            vertex_colors = np.asarray(mesh.vertex_colors)
            knn = NearestNeighbors(n_neighbors=4, algorithm='kd_tree').fit(v)
            _, indices = knn.kneighbors(query_points)
            texture = vertex_colors[indices].mean(1)

            texture = torch.tensor(texture, dtype=torch.float)
            texture = torch.clamp(texture, 0.03, 0.97).to(self.device)

        return torch.tensor(query_points).to(self.device), texture
