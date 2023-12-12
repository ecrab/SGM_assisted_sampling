import numpy as np
import numpy.random as random
from scipy.special import logsumexp

import torch.nn as nn
import torch.nn.functional as F
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch
from functorch import vmap
from functorch import grad

beta_min = 0.001
beta_max = 3

def beta_t(t):
    return beta_min + t*(beta_max - beta_min)

def alpha_t(t):
    return (t*beta_min + 0.5 * t**2 * (beta_max - beta_min))

def drift(x, t):
    return -0.5*beta_t(t)*x

def diffusivity(t):
    return np.sqrt(100*beta_t(t))

def mean_factor(t):
    return torch.exp(-0.5 * alpha_t(t))

def var(t):
    return 1 - torch.exp(-alpha_t(t))

def scan(f, init, xs, length=None):
  if xs is None:
    xs = [None] * length
  carry = init
  ys = []
  for x in xs:
    carry, y = f(carry, x)
    ys.append(y)
  return carry, np.stack(ys)

def log_hat_pt(x, t):
    N = xs.shape[0]
    means = xs * mean_factor(t)
    v = var(t)
    potentials = torch.sum(-(x - means)**2 / (2 * v), axis=1)
    lse = torch.logsumexp(potentials, 0)
    return lse

def nabla_log_hat_pt():
    return (vmap(grad(log_hat_pt), in_dims=(0, 0), out_dims=(0)))

R = 1000
train_ts = np.arange(1, R)/(R-1)

def reverse_sde(rng, N, n_samples, forward_drift, diffusivity, score, ts=train_ts):
    def f(carry, params):
        t, dt = params
        x, rng = carry
        step_rng = rng
        diff = diffusivity(1-t)
        t = np.ones((x.shape[0], 1)) * t
        x2 = torch.from_numpy(x).cpu()
        t2 = torch.from_numpy(t).cpu()
        drift = -forward_drift(x, 1-t) + diff**2 * (score(x2, 1-t2).detach().cpu().numpy())
        noise = random.normal(size=x.shape)
        x = x + dt * drift + np.sqrt(dt)*diff*noise
        return (x, rng), ()
    
    step_rng = rng
    initial = random.normal(size=(n_samples, N))
    dts = ts[1:] - ts[:-1]
    params = np.stack([ts[:-1], dts], axis=1)
    (x, _), _ = scan(f, (initial, rng), params)
    return x

