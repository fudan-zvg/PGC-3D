import os
import nvdiffrast.torch as dr
from nerf.fine.dmtet import DMTetGeometry
from nerf.fine.perspective_camera import PerspectiveCamera
from nerf.fine.neural_render import NeuralRender
from pytorch3d.loss import mesh_normal_consistency
from pytorch3d.structures import Meshes
import open3d as o3d
from nerf.fine.rast_utils import *


class Rasterizer_pbr(nn.Module):
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
        self.threshold = 99  # remove floaters
        self.ctx = dr.RasterizeCudaContext(device=self.device)

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
        texture = self.mat['kd_ks_normal'].sample(x)
        return texture

    def get_geometry_prediction(self, sdf, deformation):
        # Step 1: first get the sdf and deformation value for each vertices in the tetrahedon grid.
        v_deformed = self.dmtet_geometry.verts + self.dmtet_scale / (self.grid_res * 2) * torch.tanh(deformation)
        tets = self.dmtet_geometry.indices

        # Step 2: Using marching tet to obtain the mesh
        verts, faces, uvs, uv_idx = self.dmtet_geometry.get_mesh(
            v_deformed, sdf,
            with_uv=True, indices=tets)

        imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=self.mat)

        # Run mesh operations to generate tangent space
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)

        return verts, faces, v_deformed, imesh

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

    def export_mesh(self, path, **kwargs):
        verts = self.dmtet_geometry.verts
        geometry = self.get_geometry(verts)
        sdf = geometry[..., 0]
        delta_v = geometry[..., 1:]
        mesh_v, mesh_f, v_deformed, imesh = self.get_geometry_prediction(sdf, delta_v)

        base_mesh = xatlas_uvmap(self.ctx, imesh, self.mat, self.device,
                                 self.kd_min_max, self.ks_min_max, self.nrm_min_max)
        obj.write_obj(path, base_mesh)
        light.save_env_map(os.path.join(path, "probe.hdr"), self.lgt)
        return

    def forward(self, data, light_d=None, ratio=1, shading='albedo',
                only_points=False, query_points=None, smooth=False):

        if only_points:
            geometry = self.get_geometry(query_points)
            textures = self.get_texture(query_points)[..., :3]
            sdf = geometry[..., 0]
            return sdf, textures

        focal = data['focal']
        cam_mv = data['poses']  # Bx4x4
        H, W = data['H'], data['W']
        B = cam_mv.shape[0]

        focal = H / (2 * focal)
        dmtet_camera = PerspectiveCamera(focal=focal, device=self.device)
        self.dmtet_geometry.set_camera(dmtet_camera)

        # Generate 3D mesh first
        verts = self.dmtet_geometry.verts
        geometry = self.get_geometry(verts)
        sdf = geometry[..., 0]
        delta_v = geometry[..., 1:]
        mesh_v, mesh_f, v_deformed, imesh = self.get_geometry_prediction(sdf, delta_v)

        background = torch.ones((B, H, W, 3), dtype=torch.float32, device=self.device)
        proj_mtx = dmtet_camera.proj_mtx
        mv = torch.linalg.inv(cam_mv)
        mvp = (proj_mtx @ mv).float()
        campos = cam_mv[:, :3, 3]
        # self.lgt.build_mips()
        # self.lgt.xfm(mv)
        buffers = render.render_mesh(self.ctx, imesh, mvp, mv, campos, self.lgt, [H, W],
                                     spp=1, msaa=True, background=background, bsdf=None)

        # Render the normal into 2D image
        img_antilias = buffers['shaded'][..., 0:3]

        buffers['normals'] = -buffers['normals']

        normals_vis = (-buffers['normals'] + 1) / 2
        mask = buffers['mask']
        buffers['normals_vis'] = normals_vis * mask + (1 - mask)

        if smooth:
            mesh = Meshes(verts=[mesh_v], faces=[mesh_f])
            mesh_smooth_loss = mesh_normal_consistency(mesh)
            # mesh_smooth_loss = mesh_laplacian_smoothing(mesh)
            buffers['loss_smooth'] = mesh_smooth_loss

        # Albedo (k_d) smoothnesss regularizer
        buffers['reg_kd_smooth'] = torch.mean(buffers['kd_grad'][..., :-1] * buffers['kd_grad'][..., -1:])
        buffers['reg_ks_smooth'] = torch.mean(buffers['ks_grad'][..., :-1] * buffers['ks_grad'][..., -1:])

        # Visibility regularizer
        buffers['reg_vis'] = torch.mean(buffers['occlusion'][..., :-1] * buffers['occlusion'][..., -1:])

        # Light white balance regularizer
        buffers['reg_lgt'] = self.lgt.regularizer()

        return img_antilias, buffers

    def get_params(self, lr):
        params = [
            {'params': self.encoder.parameters(), 'lr': lr[0] * 10},
            {'params': self.geometry_MLP.parameters(), 'lr': lr[0]},
            {'params': self.mat.parameters(), 'lr': lr[1]},
            {'params': self.lgt.parameters(), 'lr': lr[1]}
        ]
        return params

    def compute_targets(self):
        if self.opt.n_targets > 0:
            geometry = self.get_geometry(self.dmtet_geometry.verts)
            sdf = geometry[..., 0]
            delta_v = geometry[..., 1:]
            mesh_v, mesh_f, v_deformed, imesh = self.get_geometry_prediction(sdf, delta_v)

            v = imesh.v_pos
            vn = imesh.v_nrm.detach().cpu().numpy()
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
        vc = np.asarray(mesh0.vertex_colors)
        # canonicalize mesh
        if np.abs(v).max() > 0.9:
            canonicalize = False
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
            texture[index] = 0.6  # avoid black area which has no change in training
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
