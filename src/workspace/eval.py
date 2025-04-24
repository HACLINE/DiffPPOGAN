from src.workspace.base import BaseWorkspace
from src.utils.util import LinearScheduler, extract_time_index, ForwardDiffusion, Sampler, update_ema
from src.utils.visualize import plot_curve

import torch
import numpy as np
import wandb
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import make_grid
from collections import OrderedDict, deque
from omegaconf import OmegaConf
import copy

class EvalWorkspace(BaseWorkspace):
    def __init__(self):
        pass

    def setup(self, cfg):
        super().setup(cfg)
        self.scheduler = LinearScheduler(cfg.beta_start, cfg.beta_end, cfg.timesteps)
        self.forward_diffusion_model = ForwardDiffusion(self.scheduler.sqrt_alphas_cumprod, self.scheduler.sqrt_one_minus_alphas_cumprod)
        self.sampler = Sampler(self.scheduler.betas, cfg.timesteps, cfg.sample_timesteps, cfg.ddpm)
        self.rollout = Sampler(self.scheduler.betas, cfg.timesteps, cfg.timesteps, cfg.ddpm)
        self.timesteps = cfg.timesteps
        self.sample_timesteps = cfg.sample_timesteps
        self.ref_timesteps = cfg.ref_timesteps