import os

import torch

from nerf.fine.dmtet import DMTetGeometry
from nerf.fine.perspective_camera import PerspectiveCamera
from nerf.fine.neural_render import NeuralRender
import cv2
from pytorch3d.loss import mesh_normal_consistency
from pytorch3d.structures import Meshes
import open3d as o3d
from encoding import get_encoder
from nerf.fine.rast_utils import *


class Rasterizer_MLP(torch.nn.Module):
    def __init__(
            self,
            opt,
            device='cuda',
            tet_res=100,  # Resolution for tetrahedron grid
            render_type='neural_render',  # neural type
            dmtet_scale=1.8,
            mesh_path=None
    ):  #
        super().__init__()
        self.opt = opt
        self.device = device
        self.dmtet_scale = dmtet_scale
        self.render_type = render_type
        self.grid_res = tet_res
        self.mesh_path = mesh_path
        self.threshold = 99.9  # remove floaters

        if self.opt.bg_scenery:
            img = cv2.imread("bg2.png", cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB) / 255
            precision_t = torch.float16 if opt.fp16 else torch.float32
            img = torch.tensor(img, dtype=precision_t, device=self.device)
            self.bg = img[None, ...]  # [1, 1024, 1024, 3]

        # Renderer we used.
        dmtet_renderer = NeuralRender(device)

        # Geometry class for DMTet
        self.dmtet_geometry = DMTetGeometry(
            grid_res=self.grid_res, bound=opt.bound, renderer=dmtet_renderer, render_type=render_type,
            device=self.device)

        # for sampling
        self.targets = None
        self.weights = None
        self.radius = None

        # for test
        self.v = None
        self.f = None

        # geometry
        self.encoder, self.in_dim = get_encoder('hashgrid', input_dim=3,
                                                log2_hashmap_size=16,
                                                desired_resolution=2 ** 12,
                                                base_resolution=2 ** 4,
                                                level_dim=4)
        self.geometry_MLP = MLP(self.in_dim, 4, 32, 3, bias=True)

        # texture
        self.encoder_tex, self.in_dim = get_encoder('hashgrid', input_dim=3,
                                                    log2_hashmap_size=16,
                                                    desired_resolution=2 ** 12,
                                                    base_resolution=2 ** 4,
                                                    level_dim=4)
        out_dim = 4 if self.opt.latent else 3
        self.texture_MLP = MLP(self.in_dim, out_dim, 32, 2, bias=True)

        # init geometry
        points, sdf, texture = self.init_geometry(canonicalize=opt.canonicalize)
        self.points = points
        self.sdf = sdf
        self.texture = texture

    def get_geometry(self, x):
        hidden = self.encoder(x, bound=self.opt.bound)
        geometry = self.geometry_MLP(hidden)
        return geometry

    def get_texture(self, x):
        hidden = self.encoder_tex(x, bound=self.opt.bound)
        texture = self.texture_MLP(hidden)
        if not self.opt.latent:
            texture = torch.sigmoid(texture) * 0.9 + 0.05  # [0.05, 0.95]
        return texture

    def get_geometry_prediction(self, sdf, deformation):
        # Step 1: first get the sdf and deformation value for each vertices in the tetrahedon grid.
        v_deformed = self.dmtet_geometry.verts + self.dmtet_scale / (self.grid_res * 2) * torch.tanh(deformation)
        tets = self.dmtet_geometry.indices

        # Step 2: Using marching tet to obtain the mesh
        verts, faces = self.dmtet_geometry.get_mesh(
            v_deformed, sdf,
            with_uv=False, indices=tets)

        return verts, faces, v_deformed

    def render_mesh(self, mesh_v, mesh_f, mesh_vn, textures, cam_mv, resolution=512):
        return_value = self.dmtet_geometry.render_mesh(
            mesh_v,
            mesh_f.int(),
            mesh_vn,
            textures,
            cam_mv,
            resolution=resolution,
            hierarchical_mask=False
        )

        return return_value

    def export_mesh(self, path, name='', type='vertex_color', remove_floaters=True, SD=None, flip=True):
        if name == "init":
            remove_floaters = False
            type = 'vertex_color'

        verts = self.dmtet_geometry.verts
        geometry = self.get_geometry(verts)
        sdf = geometry[..., 0]
        delta_v = geometry[..., 1:]
        mesh_v, mesh_f, v_deformed = self.get_geometry_prediction(sdf, delta_v)

        # Step 2: texture
        # texture = self.get_texture(verts)
        # tets = self.dmtet_geometry.indices
        # textures, _ = self.dmtet_geometry.get_mesh(
        #     texture, sdf,
        #     with_uv=False, indices=tets)

        mesh = o3d.geometry.TriangleMesh()
        if flip:
            mesh_v[:, 0] = -mesh_v[:, 0]
        vertex = mesh_v.detach().cpu().numpy()
        mesh.vertices = o3d.utility.Vector3dVector(vertex)
        mesh.triangles = o3d.utility.Vector3iVector(mesh_f.int().detach().cpu().numpy())
        # mesh.vertex_colors = o3d.utility.Vector3dVector(textures.detach().cpu().numpy())

        if remove_floaters:
            print("removing floaters...")
            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
            triangle_clusters = np.asarray(triangle_clusters)
            cluster_n_triangles = np.asarray(cluster_n_triangles)

            threshold = np.percentile(cluster_n_triangles, self.threshold)
            triangles_to_remove = cluster_n_triangles[triangle_clusters] < threshold
            mesh.remove_triangles_by_mask(triangles_to_remove)

        # o3d.io.write_triangle_mesh(os.path.join(path, name + f'fine_mesh_vc.obj'), mesh)

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

            if self.opt.latent:
                h, w = 128, 128

            rast, _ = dr.rasterize(glctx, uv.unsqueeze(0), ft, (h, w))  # [1, h, w, 4]
            xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f)  # [1, h, w, 3]
            mask, _ = dr.interpolate(torch.ones_like(v[:, :1]).unsqueeze(0), rast, f)  # [1, h, w, 1]

            # masked query
            xyzs = xyzs.view(-1, 3)
            mask = (mask > 0).view(-1)

            dim = 4 if self.opt.latent else 3
            feats = torch.zeros(h * w, dim, device=device, dtype=torch.float32)

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

            if self.opt.latent:
                feats = feats[None, ...]  # [1, h, w, 4]
                feats = SD.decode_latents(feats.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)[0]

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

        if type != 'vertex_color':
            device = mesh_v.device
            mesh_v = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32).to(device)
            mesh_v[:, 0] = -mesh_v[:, 0]
            mesh_f = torch.tensor(np.asarray(mesh.triangles), dtype=torch.float32).to(device)
            _export(mesh_v, mesh_f.int())
        return

    def forward(self, data, light_d=None, ratio=1, shading='albedo',
                only_points=False, query_points=None,
                smooth=False, eval=False):

        if only_points:
            geometry = self.get_geometry(query_points)
            textures = self.get_texture(query_points)
            sdf = geometry[..., 0]
            return sdf, textures

        focal = data['focal']
        cam_mv = data['poses']  # Bx4x4
        rays_o = data['rays_o']
        rays_d = data['rays_d']
        H, W = data['H'], data['W']
        B = rays_o.shape[0]

        focal = H / (2 * focal)
        dmtet_camera = PerspectiveCamera(focal=focal, device=self.device)
        self.dmtet_geometry.set_camera(dmtet_camera)

        # Generate 3D mesh first
        if self.opt.test:
            if self.v is None:
                verts = self.dmtet_geometry.verts
                geometry = self.get_geometry(verts)
                sdf = geometry[..., 0]
                delta_v = geometry[..., 1:]
                mesh_v, mesh_f, v_deformed = self.get_geometry_prediction(sdf, delta_v)
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(mesh_v.detach().cpu().numpy())
                mesh.triangles = o3d.utility.Vector3iVector(mesh_f.int().detach().cpu().numpy())
                with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
                triangle_clusters = np.asarray(triangle_clusters)
                cluster_n_triangles = np.asarray(cluster_n_triangles)
                threshold = np.percentile(cluster_n_triangles, self.threshold)
                triangles_to_remove = cluster_n_triangles[triangle_clusters] < threshold
                mesh.remove_triangles_by_mask(triangles_to_remove)
                mesh_v = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32).to(self.device)
                mesh_f = torch.tensor(np.asarray(mesh.triangles), dtype=torch.int32).to(self.device)
                self.v = mesh_v
                self.f = mesh_f
            else:
                mesh_v = self.v
                mesh_f = self.f
        else:
            verts = self.dmtet_geometry.verts
            geometry = self.get_geometry(verts)
            sdf = geometry[..., 0]
            delta_v = geometry[..., 1:]
            mesh_v, mesh_f, v_deformed = self.get_geometry_prediction(sdf, delta_v)
        mesh = Meshes(verts=[mesh_v], faces=[mesh_f])
        mesh_vn = mesh.verts_normals_packed()

        # Render the normal into 2D image
        return_value = self.render_mesh(mesh_v, mesh_f, mesh_vn, mesh_v, cam_mv, resolution=H)
        tex_pos = return_value['tex_pos']
        antilias_mask = return_value['mask']
        normals = -return_value['normals']

        # texture
        albedo = self.get_texture(tex_pos)

        # Predict the RGB color for each pixel
        if shading == 'albedo':
            img = albedo
        else:
            # random sample light_d if not provided
            if light_d is None:
                # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
                light_d = (rays_o + torch.randn(3, device=self.device, dtype=torch.float))
                light_d = safe_normalize(light_d).reshape(B, H, W, 3)

            # lambertian shading
            lambertian = ratio + (1 - ratio) * (normals[:B] * light_d).sum(-1).clamp(min=0)  # [B, H, W]

            if shading == 'normal':
                img = (normals + 1) / 2
            else:  # 'lambertian'
                img = albedo[:B] * lambertian.unsqueeze(-1)
                img = torch.cat([img, albedo[B:]], dim=0)

        # background
        if self.opt.bg_scenery and not eval:
            bg_color = F.interpolate(self.bg.permute(0, 3, 1, 2),
                                     (H, W), mode='bilinear', align_corners=False)
            bg_color = bg_color.permute(0, 2, 3, 1)
        else:
            bg_color = self.opt.bg_color

        img_antilias = img * antilias_mask + bg_color * (1 - antilias_mask)

        if smooth:
            mesh_smooth_loss = mesh_normal_consistency(mesh)
            # mesh_smooth_loss += mesh_laplacian_smoothing(mesh)
            return_value['loss_smooth'] = mesh_smooth_loss

        normal_map = (normals + 1) / 2
        normal_map = normal_map * antilias_mask + (1 - antilias_mask)
        return_value['normals_vis'] = normal_map

        return img_antilias, return_value

    def get_params(self, lr):
        params = [
            {'params': self.encoder.parameters(), 'lr': lr[0] * 10},
            {'params': self.geometry_MLP.parameters(), 'lr': lr[0]},
            {'params': self.encoder_tex.parameters(), 'lr': lr[1] * 10},
            {'params': self.texture_MLP.parameters(), 'lr': lr[1]},
        ]

        return params

    def compute_targets(self):
        if self.opt.n_targets > 0:
            verts = self.dmtet_geometry.verts
            geometry = self.get_geometry(verts)
            sdf = geometry[..., 0]
            delta_v = geometry[..., 1:]
            mesh_v, mesh_f, v_deformed = self.get_geometry_prediction(sdf, delta_v)
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(mesh_v.detach().cpu().numpy())
            mesh.triangles = o3d.utility.Vector3iVector(mesh_f.int().detach().cpu().numpy())

            # texture
            tets = self.dmtet_geometry.indices
            texture = self.get_texture(verts)
            textures, _ = self.dmtet_geometry.get_mesh(
                texture, sdf,
                with_uv=False, indices=tets)
            vc = textures.detach().cpu().numpy()

            mesh.compute_vertex_normals()
            v = np.asarray(mesh.vertices)
            vn = np.asarray(mesh.vertex_normals)
            targets, weights, radius, _ = compute_targets(v, vn, self.opt.n_targets, vc)
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

            threshold = np.percentile(cluster_n_triangles, 90)
            triangles_to_remove = cluster_n_triangles[triangle_clusters] < threshold
            mesh0.remove_triangles_by_mask(triangles_to_remove)

        v = np.asarray(mesh0.vertices)
        f = np.asarray(mesh0.triangles)
        vc = np.asarray(mesh0.vertex_colors)
        # canonicalize mesh
        # if np.abs(v).max() > 0.9:
        #     canonicalize = False
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

            dtype = self.dmtet_geometry.verts.dtype
            texture = torch.tensor(texture, dtype=dtype)
            texture = torch.clamp(texture, 1e-3, 1 - 1e-3).to(self.device)

            index = texture.abs().sum(-1) < 0.5
            texture[index] = 0.2  # avoid black area which has no change in training
            index = texture.abs().sum(-1) > 2.5
            texture[index] = 0.8  # avoid white area which become background

        # sdf
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        # Create a scene and add the triangle mesh
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)
        sdf = scene.compute_signed_distance(query_points)
        sdf = sdf.numpy()

        return torch.tensor(query_points).to(self.device), torch.tensor(sdf).to(self.device), texture


if __name__ == '__main__':
    base = '/SSD_DISK/users/jiangjunzhe/dreamfusion_fine/df_fine/exp/fox/ls/mesh/'
    path = base + 'fine_mesh_vc.obj'
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)
    vn = np.asarray(mesh.vertex_normals)
    vc = np.asarray(mesh.vertex_colors)
    targets, _, radius, weights = compute_targets(v, vn, 4, vc)
    print(targets)
    print(radius)

    from matplotlib import cm

    colormap = cm.get_cmap('turbo')
    curve_fn = lambda x: np.log(x + np.finfo(np.float32).eps)
    eps = np.finfo(np.float32).eps
    near = weights.min() + eps
    far = weights.max() - eps
    near, far, weights = [curve_fn(x) for x in [near, far, weights]]
    weights = np.nan_to_num(
        np.clip((weights - np.minimum(near, far)) / np.abs(far - near), 0, 1))
    vis = colormap(weights)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vis[:, :3])
    o3d.io.write_triangle_mesh(os.path.join(path, base + 'mesh_weights.obj'), mesh)

    v_norm = np.sqrt((v ** 2).sum(1))
    unit_v = v / v_norm[..., None]
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(unit_v)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vis[:, :3])
    o3d.io.write_triangle_mesh(os.path.join(path, base + 'sphere_weights.obj'), mesh)
