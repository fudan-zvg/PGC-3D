import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture

import xatlas
from encoding import get_encoder

import others.nvdiffrec.render.renderutils as ru
from others.nvdiffrec.render import obj
from others.nvdiffrec.render import material
from others.nvdiffrec.render import util
from others.nvdiffrec.render import mesh
from others.nvdiffrec.render import texture
#from others.nvdiffrec.render import mlptexture
from others.nvdiffrec.render import light
from others.nvdiffrec.render import render


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden,
                                 self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x


class Material_mlp(nn.Module):
    def __init__(self, min_max):
        super().__init__()
        self.min_max = min_max
        self.encoder, self.in_dim = get_encoder('hashgrid', input_dim=3,
                                                log2_hashmap_size=16,
                                                desired_resolution=2 ** 12,
                                                base_resolution=2 ** 4,
                                                level_dim=4)
        self.texture_MLP = MLP(self.in_dim, 9, 32, 2, bias=True)

    def sample(self, texc):
        _texc = texc.view(-1, 3)
        p_enc = self.encoder(_texc.contiguous())
        out = self.texture_MLP(p_enc)

        # Sigmoid limit and scale to the allowed range
        out = torch.sigmoid(out) * (self.min_max[1][None, :] - self.min_max[0][None, :]) + self.min_max[0][None, :]

        return out.view(*texc.shape[:-1], 9)


def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


def compute_targets(v, vn, n_targets, vc=None):
    knn = NearestNeighbors(n_neighbors=20, algorithm='kd_tree').fit(v)
    _, indices = knn.kneighbors(v)
    vn_neighbors = vn[indices]
    if vc is not None:
        vc_neighbors = vc[indices]

    v_num = v.shape[0]
    normal_weights = np.ones(v_num)
    rgb_weights = np.ones(v_num)
    for i in range(v_num):
        normal_weights[i] = np.linalg.norm(np.cov(vn_neighbors[i].T), 'fro')
        if vc is not None:
            rgb_weights[i] = np.linalg.norm(np.cov(vc_neighbors[i].T), 'fro')

    v_norm = np.sqrt((v ** 2).sum(1))
    unit_v = v / v_norm[..., None]
    k = n_targets
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=0)
    gmm.fit(unit_v)
    pred_label = gmm.predict(unit_v)
    pred_prob = gmm.predict_proba(unit_v).max(1)
    weights_ = pred_prob * (v_norm ** 4) * normal_weights * rgb_weights
    targets = np.zeros([k, 3])
    radius = np.ones([k])
    weights = gmm.weights_
    for i in range(k):
        index = pred_label == i
        w_sum = weights_[index].sum()
        targets[i] = (v[index] * weights_[index][..., None]).sum(0) / w_sum
        radius[i] = np.quantile(np.sqrt(((v[index] - targets[i]) ** 2).sum(1)), 0.3)
        weights[i] = weights[i] * w_sum

    weights = weights / weights.sum()
    return targets, weights, radius, weights_


# for save mesh
@torch.no_grad()
def xatlas_uvmap(glctx, eval_mesh, mat, device,
                 kd_min_max, ks_min_max, nrm_min_max):

    # Create uvs with xatlas
    v_pos = eval_mesh.v_pos.detach().cpu().numpy()
    t_pos_idx = eval_mesh.t_pos_idx.detach().cpu().numpy()
    vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)

    # Convert to tensors
    indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)

    uvs = torch.tensor(uvs, dtype=torch.float32, device=device)
    faces = torch.tensor(indices_int64, dtype=torch.int64, device=device)

    new_mesh = mesh.Mesh(v_tex=uvs, t_tex_idx=faces, base=eval_mesh)

    mask, kd, ks, normal = render.render_uv(glctx, new_mesh, [2048, 2048], eval_mesh.material['kd_ks_normal'])

    # if FLAGS.layers > 1:
    #     kd = torch.cat((kd, torch.rand_like(kd[..., 0:1])), dim=-1)

    new_mesh.material = material.Material({
        'bsdf': mat['bsdf'],
        'kd': texture.Texture2D(kd, min_max=kd_min_max),
        'ks': texture.Texture2D(ks, min_max=ks_min_max),
        'normal': texture.Texture2D(normal, min_max=nrm_min_max),
        'no_perturbed_nrm': mat['no_perturbed_nrm']
    })

    return new_mesh
