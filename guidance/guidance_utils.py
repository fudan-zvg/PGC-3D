import torch
from torch.cuda.amp import custom_bwd, custom_fwd
import numpy as np


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def w_star(t, m1=800, m2=500, s1=300, s2=100):
    # max time 1000
    r = np.ones_like(t) * 1.0
    r[t > m1] = np.exp(-((t[t > m1] - m1) ** 2) / (2 * s1 * s1))
    r[t < m2] = np.exp(-((t[t < m2] - m2) ** 2) / (2 * s2 * s2))
    return r


def precompute_prior(T=1000, min_t=200, max_t=800):
    ts = np.arange(T)
    prior = w_star(ts)[min_t:max_t]
    prior = prior / prior.sum()
    prior = prior[::-1].cumsum()[::-1]
    return prior, min_t


def time_prioritize(step_ratio, time_prior, min_t=200):
    return np.abs(time_prior - step_ratio).argmin() + min_t


def noise_norm(eps):
    # [B, 3, H, W]
    return torch.sqrt(torch.square(eps).sum(dim=[1, 2, 3]))


if __name__ == '__main__':
    prior, _ = precompute_prior()
    t = time_prioritize(0.5, prior)


