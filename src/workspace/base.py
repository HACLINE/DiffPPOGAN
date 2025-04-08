from src.models import get_model
from src.data.Dataloaders import pick_dataset
from src.utils.util import LinearScheduler, extract_time_index, ForwardDiffusion

import torch
from diffusers import AutoencoderKL
import copy
import os
import numpy as np
from tqdm import tqdm
import cv2
from pathlib import Path
from pytorch_fid import fid_score
from datetime import datetime

class BaseWorkspace:
    def __init__(self):
        pass

    def setup(self, cfg):
        self.cfg = cfg
        self.device = torch.device(f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() and cfg.gpu_id >= 0 else "cpu")
        self.tune_timesteps = cfg.tune_timesteps
        self.setmodel(cfg, self.device)
        self.setdataset(cfg, self.device)

        self.vae =  AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(self.device) if cfg.latent else None
        self.channels = cfg.channels
        self.img_size = cfg.image_size
        self.dataset = cfg.dataset.name

        self.model_update_epoch = 0
        self.wandb_config = cfg.wandb

        self.ema = copy.deepcopy(self.model)
        self.ema_rate = cfg.ema_rate
        for param in self.ema.parameters():
            param.requires_grad = False

        self.cur_time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cfg.cur_time_string = self.cur_time_string
        self.cfg = cfg

        fid_cfg = cfg.fid
        self.real_image_path = Path(fid_cfg.real_image_path)
        self.eval_fid_every = fid_cfg.eval_fid_every
        self.eval_fid_num = fid_cfg.eval_fid_num
        self.eval_fid_steps = fid_cfg.eval_fid_steps
        self.eval_fid_batch_size = fid_cfg.eval_fid_batch_size

        if self.eval_fid_every > 0:
            assert self.real_image_path is not None, "Please provide the path to the real images for FID calculation"
            block_idx = fid_score.InceptionV3.BLOCK_INDEX_BY_DIM[2048]

            self.inception_model = fid_score.InceptionV3([block_idx]).to(self.device)

            if (self.real_image_path.parent / f"statistics.pt").exists():
                self.real_m, self.real_s = torch.load(self.real_image_path.parent / f"statistics.pt", weights_only=False)
            else:
                self.real_m, self.real_s = fid_score.compute_statistics_of_path(self.real_image_path, self.inception_model, 64, 2048, self.device, 8)
                torch.save((self.real_m, self.real_s), self.real_image_path.parent / f"statistics.pt")

        self.models_dir = Path("models") / self.cur_time_string
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def load_checkpoint(self, checkpoint_path):
        '''
        Load the model checkpoint
        :param checkpoint_path: path to the checkpoint
        '''
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.eval()

    def setmodel(self, cfg, device):
        self.model = get_model(cfg.model).to(device)
        self.model.load_state_dict(torch.load(cfg.ref_model_path))
        self.model.store_ref_model(copy.deepcopy(self.model).eval())

        self.critic = get_model(cfg.critic).to(device)
        self.discriminator = get_model(cfg.discriminator).to(device)
        
    def setdataset(self, cfg, device):
        dataset_cfg = cfg.dataset
        self.dataloader, _, _ = pick_dataset(dataset_cfg.name, 'train', cfg.batch_size, normalize=dataset_cfg.normalize, size=dataset_cfg.size, num_workers=dataset_cfg.num_workers)

    @torch.no_grad()
    def fid_sample(self, batch_size=16):
        '''
        Sample images for FID calculation
        :param batch_size: batch size
        '''
        self.load_checkpoint(self.cfg.checkpoint)
        if 'epoch' in self.cfg.checkpoint:
            ep = int(self.cfg.checkpoint.split('epoch')[1].split('.')[0])
        else:
            ep = 0

        model_type = "ema" if "ema" in self.cfg.checkpoint else "model"

        date = self.cfg.checkpoint.split('/')[-2]
        path = Path("fid_samples") / self.dataset / f"{date}_timesteps_{self.cfg.sample_timesteps}_ep{ep}_{model_type}"
        path.mkdir(parents=True, exist_ok=True)
        cnt = 0
        for i in tqdm(range(50000//batch_size), desc='FID Sampling', leave=True):
            samps = self.sampler.sample(model=self.model, image_size=self.img_size, batch_size=batch_size, channels=self.channels)[-1]

            if self.vae is not None:
                samps = torch.tensor(samps, device=self.device)
                samps = self.vae.decode(samps / 0.18215).sample 
                samps = samps.cpu().detach().numpy()

            samps = samps * 0.5 + 0.5
            samps = samps.clip(0, 1)
            samps = samps.transpose(0,2,3,1)
            samps = (samps*255).astype(np.uint8)
            for samp in samps:
                cv2.imwrite(path / f"{cnt}.png", cv2.cvtColor(samp, cv2.COLOR_RGB2BGR) if samp.shape[-1] == 3 else samp)
                cnt += 1  

    @torch.no_grad()
    def fid_5k_sample(self, batch_size=16):
        '''
        Sample images for FID calculation
        :param batch_size: batch size
        '''
        self.load_checkpoint(self.cfg.checkpoint)
        if 'epoch' in self.cfg.checkpoint:
            ep = int(self.cfg.checkpoint.split('epoch')[1].split('.')[0])
        else:
            ep = 0

        model_type = "ema" if "ema" in self.cfg.checkpoint else "model"

        date = self.cfg.checkpoint.split('/')[-2]
        path = Path("fid_5k_samples") / self.dataset / f"{date}_timesteps_{self.cfg.sample_timesteps}_ep{ep}_{model_type}"
        path.mkdir(parents=True, exist_ok=True)
        cnt = 0
        for i in tqdm(range(5000//batch_size), desc='FID Sampling', leave=True):
            samps = self.sampler.sample(model=self.model, image_size=self.img_size, batch_size=batch_size, channels=self.channels)[-1]

            if self.vae is not None:
                samps = torch.tensor(samps, device=self.device)
                samps = self.vae.decode(samps / 0.18215).sample 
                samps = samps.cpu().detach().numpy()

            samps = samps * 0.5 + 0.5
            samps = samps.clip(0, 1)
            samps = samps.transpose(0,2,3,1)
            samps = (samps*255).astype(np.uint8)
            for samp in samps:
                cv2.imwrite(path / f"{cnt}.png", cv2.cvtColor(samp, cv2.COLOR_RGB2BGR) if samp.shape[-1] == 3 else samp)
                cnt += 1  

    @torch.no_grad()
    def eval_fid(self):
        '''
        Evaluate FID
        :param batch_size: batch size
        '''
        if self.vae is not None:
            self.vae.eval()
        self.model.eval()

        date = self.cur_time_string
        path = Path("fid_tmp") / self.dataset / f"{date}_timesteps_{self.eval_fid_steps}_ep{self.epoch}_model"
        path.mkdir(parents=True, exist_ok=True)
        cnt = 0

        self.sampler.set_sample_timesteps(self.eval_fid_steps)

        for i in tqdm(range(self.eval_fid_num//self.eval_fid_batch_size), desc='FID Sampling', leave=True):
            samps = self.sampler.sample(model=self.model, image_size=self.img_size, batch_size=self.eval_fid_batch_size, channels=self.channels)[-1]

            if self.vae is not None:
                samps = torch.tensor(samps, device=self.device)
                samps = self.vae.decode(samps / 0.18215).sample 
                samps = samps.cpu().detach().numpy()

            samps = samps * 0.5 + 0.5
            samps = samps.clip(0, 1)
            samps = samps.transpose(0,2,3,1)
            samps = (samps*255).astype(np.uint8)
            for samp in samps:
                cv2.imwrite(path / f"{cnt}.png", cv2.cvtColor(samp, cv2.COLOR_RGB2BGR) if samp.shape[-1] == 3 else samp)
                cnt += 1
        
        self.sampler.set_sample_timesteps(self.sample_timesteps)
        
        fake_m, fake_s = fid_score.compute_statistics_of_path(str(path), self.inception_model, 64, 2048, self.device, 8)
        fid_value = fid_score.calculate_frechet_distance(fake_m, fake_s, self.real_m, self.real_s)

        return fid_value
    
    @torch.no_grad()
    def eval_ema_fid(self):
        '''
        Evaluate FID
        :param batch_size: batch size
        '''
        if self.vae is not None:
            self.vae.eval()
        self.ema.eval()

        date = self.cur_time_string
        path = Path("fid_tmp") / self.dataset / f"{date}_timesteps_{self.eval_fid_steps}_ep{self.epoch}_ema"
        path.mkdir(parents=True, exist_ok=True)
        cnt = 0

        self.sampler.set_sample_timesteps(self.eval_fid_steps)

        for i in tqdm(range(self.eval_fid_num//self.eval_fid_batch_size), desc='FID Sampling', leave=True):
            samps = self.sampler.sample(model=self.ema, image_size=self.img_size, batch_size=self.eval_fid_batch_size, channels=self.channels)[-1]

            if self.vae is not None:
                samps = torch.tensor(samps, device=self.device)
                samps = self.vae.decode(samps / 0.18215).sample 
                samps = samps.cpu().detach().numpy()

            samps = samps * 0.5 + 0.5
            samps = samps.clip(0, 1)
            samps = samps.transpose(0,2,3,1)
            samps = (samps*255).astype(np.uint8)
            for samp in samps:
                cv2.imwrite(path / f"{cnt}.png", cv2.cvtColor(samp, cv2.COLOR_RGB2BGR) if samp.shape[-1] == 3 else samp)
                cnt += 1
        
        self.sampler.set_sample_timesteps(self.sample_timesteps)
        
        fake_m, fake_s = fid_score.compute_statistics_of_path(str(path), self.inception_model, 64, 2048, self.device, 8)
        fid_value = fid_score.calculate_frechet_distance(fake_m, fake_s, self.real_m, self.real_s)

        return fid_value