import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict

class LinearScheduler():
    def __init__(self, beta_start=0.0001, beta_end=0.02, timesteps=1000):
        '''
        Linear scheduler
        :param beta_start: starting beta value
        :param beta_end: ending beta value
        :param timesteps: number of timesteps
        '''
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.betas = self._linear_beta_schedule()
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_one_by_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod = self._compute_forward_diffusion_alphas(self.alphas_cumprod)
        self.posterior_variance = self._compute_posterior_variance(self.alphas_cumprod_prev, self.alphas_cumprod)

    def _compute_forward_diffusion_alphas(self, alphas_cumprod):
        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod
    
    def _compute_posterior_variance(self, alphas_cumprod_prev, alphas_cumprod):
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        return self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    def _linear_beta_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.timesteps)  

def extract_time_index(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
class ForwardDiffusion():
    def __init__(self, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
        '''
        Forward diffusion module
        :param sqrt_alphas_cumprod: square root of the cumulative product of alphas
        :param sqrt_one_minus_alphas_cumprod: square root of the cumulative product of 1 - alphas
        '''
        self.sqrt_alphas_cumprod = sqrt_alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract_time_index(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract_time_index(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

class Sampler():
    def __init__(self, betas, timesteps=1000, sample_timesteps=100, ddpm=1.0):
        '''
        Sampler module
        :param betas: beta values
        :param timesteps: number of timesteps
        :param sample_timesteps: number of sample timesteps
        :param ddpm: diffusion coefficient
        :param recon_factor: starts reconstruction at recon_factor * timesteps
        '''
        self.betas = betas
        self.alphas = (1-self.betas).cumprod(dim=0)
        self.timesteps = timesteps
        self.sample_timesteps = sample_timesteps
        self.ddpm = ddpm
        self.scaling = timesteps//sample_timesteps

    def set_sample_timesteps(self, sample_timesteps):
        '''
        Set sample timesteps
        :param sample_timesteps: number of sample timesteps
        '''
        self.sample_timesteps = sample_timesteps
        self.scaling = self.timesteps//sample_timesteps
    
    @torch.no_grad()
    def p_sample(self, model, x, t, tau_index):
        '''
        Sample from the model
        :param model: model
        :param x: input image
        :param t: time
        :param tau_index: tau index
        '''
        alpha_t = extract_time_index(self.alphas, t, x.shape).to(x.device)
        x0_t = (x - (1-alpha_t).sqrt()*model(x, t))/alpha_t.sqrt()

        if tau_index == (self.scaling-1): # last step
            return x0_t
        else:
            alpha_prev_t = extract_time_index(self.alphas, t-self.scaling, x.shape)
            c1 = self.ddpm*((1 - alpha_t/alpha_prev_t) * (1-alpha_prev_t) / (1 - alpha_t)).sqrt()
            c2  = ((1-alpha_prev_t) - c1**2).sqrt()
            noise = torch.randn_like(x)
            return x0_t*alpha_prev_t.sqrt() + c2*model(x,t) +  c1* noise

    @torch.no_grad()
    def p_action_sample(self, action, x, t, tau_index):
        '''
        Sample from the model
        :param action: action
        :param x: input image
        :param tau_index: tau index
        '''
        alpha_t = extract_time_index(self.alphas, t, x.shape)
        x0_t = (x - (1-alpha_t).sqrt()*action)/alpha_t.sqrt()

        if tau_index == (self.scaling-1): # last step
            return x0_t
        else:
            alpha_prev_t = extract_time_index(self.alphas, t-self.scaling, x.shape)
            c1 = self.ddpm*((1 - alpha_t/alpha_prev_t) * (1-alpha_prev_t) / (1 - alpha_t)).sqrt()
            c2  = ((1-alpha_prev_t) - c1**2).sqrt()
            noise = torch.randn_like(x)
            return x0_t*alpha_prev_t.sqrt() + c2*action +  c1* noise
    
    @torch.no_grad()
    def  p_sample_loop(self, model, shape):
        '''
        Sample from the model
        :param model: model
        :param shape: shape of the input image
        '''
        device = next(model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(range(self.timesteps-1,-1,-self.scaling), desc="Sampling", leave=False):
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())
        return imgs
        
        '''
        for i in tqdm(range(self.sample_timesteps-1,-1,-1), desc="Sampling", leave=False):
            scaled_i = i*self.scaling
            img = self.p_sample(model, img, torch.full((b,), scaled_i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())
        return imgs
        '''

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3):
        '''
        Sample from the model
        :param model: model
        :param image_size: size of the image
        :param batch_size: batch size
        :param channels: number of channels
        '''
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

@torch.no_grad()
def update_ema(ema_model, model, decay=0.5):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # if name contains "module" then remove module
        if "module" in name:
            name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
