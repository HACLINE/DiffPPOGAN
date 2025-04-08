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

class DiffPPOGANBetterAdvWorkspace(BaseWorkspace):
    def __init__(self):
        pass

    def setup(self, cfg):
        super().setup(cfg)
        self.scheduler = LinearScheduler(cfg.beta_start, cfg.beta_end, cfg.timesteps)
        self.forward_diffusion_model = ForwardDiffusion(self.scheduler.sqrt_alphas_cumprod, self.scheduler.sqrt_one_minus_alphas_cumprod)
        self.sampler = Sampler(self.scheduler.betas, cfg.timesteps, cfg.sample_timesteps, cfg.ddpm)
        self.ref_rollout = Sampler(self.scheduler.betas, cfg.timesteps, cfg.ref_sample_timesteps, cfg.ddpm)
        self.rollout = Sampler(self.scheduler.betas, cfg.timesteps, cfg.timesteps, cfg.ddpm)
        self.n_epochs = cfg.n_epochs
        self.timesteps = cfg.timesteps
        self.ref_timesteps = cfg.ref_timesteps
        self.ref_sample_timesteps = cfg.ref_sample_timesteps
        assert (self.timesteps % self.ref_sample_timesteps) == 0
        assert (self.ref_sample_timesteps % (self.timesteps // self.ref_sample_timesteps)) == 0
        self.tune_timesteps = self.timesteps - self.ref_timesteps
        self.sample_and_save_freq = cfg.sample_and_save_freq
        self.lr = cfg.lr
        self.decay = cfg.decay
        self.snapshot = cfg.n_epochs//cfg.snapshots if cfg.snapshot_per_epochs == 0 else cfg.snapshot_per_epochs
        self.sample_timesteps = cfg.sample_timesteps

        self.tune_timesteps_cdf = self.scheduler.alphas_cumprod
        self.reward_schedule_decay = cfg.reward_schedule_decay
        self.reward_schedule = self.scheduler.alphas_cumprod
        self.reward_schedule_step = torch.pow(self.reward_schedule, self.reward_schedule_decay)
        self.total_reward_weight = self.reward_schedule[:self.tune_timesteps].sum()
        self.discriminator_schedule = self.scheduler.alphas_cumprod
        # self.discriminator_schedule = torch.ones(self.timesteps)
        self.total_discriminator_weight = self.discriminator_schedule[:self.tune_timesteps].sum()
        # self.discriminator_schedule_step = torch.ones(self.timesteps)
        self.discriminator_discount = torch.tensor([np.power(cfg.discount, -i) for i in range(self.timesteps)])

        self.reward_max = cfg.reward_max
        self.reward_min = cfg.reward_min
        self.ddpm_start = cfg.ddpm_start
        self.ddpm_end = cfg.ddpm_end

        self.lr_discriminator = cfg.lr_discriminator
        self.batch_size = cfg.batch_size
        self.sample_batch_size = cfg.sample_batch_size

        self.discount = cfg.discount
        self.lr_critic = cfg.lr_critic
        self.clip = cfg.clip
        self.gae_lambda = cfg.gae_lambda
        self.std = cfg.std
        self.std_ddpm = None
        self.k_ddpm = None
        self.clip_coef = cfg.clip_coef
        self.norm_adv = cfg.norm_adv
        self.clip_vloss = cfg.clip_vloss
        self.entropy_coef = cfg.entropy_coef
        self.vf_coef = cfg.vf_coef
        self.target_kl = cfg.target_kl
        self.kl_coef = cfg.kl_coef
        self.n_epochs_train_only_value = cfg.n_epochs_train_only_value
        self.real_fake_gap = cfg.real_fake_gap
        

    def update_schedule(self):
        self.reward_schedule = self.reward_schedule * self.reward_schedule_step
        self.total_reward_weight = self.reward_schedule[:self.tune_timesteps].sum()
        self.reward_cumsum = self.reward_schedule.clone()
        for i in range(1, self.tune_timesteps):
            self.reward_cumsum[i] += self.reward_cumsum[i - 1] * self.discount
        self.reversed_reward_cumsum = self.reward_cumsum[:self.tune_timesteps].flip(0).to(self.device)
        self.discriminator_schedule = np.power(self.discount, self.tune_timesteps) * self.reward_schedule * self.discriminator_discount
        # self.discriminator_schedule = self.reward_schedule
        self.total_discriminator_weight = self.discriminator_schedule[:self.tune_timesteps].sum()

    def set_std_schedule(self, ddpm):
        base = np.sqrt((1 - self.scheduler.alphas_cumprod) / self.scheduler.alphas) - np.sqrt(1 - self.scheduler.alphas_cumprod_prev)
        sigma = ddpm * np.sqrt((1 - self.scheduler.alphas) * (1 - self.scheduler.alphas_cumprod_prev) / (1 - self.scheduler.alphas_cumprod))
        k = np.sqrt((1 - self.scheduler.alphas_cumprod) / self.scheduler.alphas) - np.sqrt(1 - self.scheduler.alphas_cumprod_prev - np.square(sigma))
        self.std_ddpm = (sigma / base).to(self.device)
        self.k_ddpm = (k / base).to(self.device)
        
    @torch.no_grad()
    def calculate_advantage(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
    ):
        advantages = torch.zeros_like(values)
        lastgaelam = 0
        for t in reversed(range(self.tune_timesteps)):
            if t == self.tune_timesteps - 1:
                delta = rewards[t] - values[t]
                advantages[t] = lastgaelam = (
                    delta
                )
            else:
                delta = self.discount * values[t + 1] - values[t] + rewards[t]
                advantages[t] = lastgaelam = (
                    delta + self.discount * self.gae_lambda * lastgaelam
                )
        returns = advantages + values
        # advantages = advantages / self.reversed_reward_cumsum.reshape(-1, 1).expand(self.tune_timesteps, self.sample_batch_size)
        advantages = (advantages - advantages.mean(dim=1, keepdim=True)) / (advantages.std(dim=1, keepdim=True) + 1e-8)
        return advantages, returns

    def train_model(self):
        wandb.init(
            project=self.wandb_config.project,
            name=self.wandb_config.name,
            config=OmegaConf.to_container(self.cfg, resolve=True),
            mode=self.wandb_config.mode,
        )

        adversarial_loss = torch.nn.BCELoss(reduction='none')
        valid = torch.ones(self.batch_size, 1).to(self.device)
        fake = torch.zeros(self.batch_size, 1).to(self.device)

        best_loss = np.inf
        epoch_bar = tqdm(range(self.n_epochs), desc='Epochs', leave=True)

        model_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.decay)
        discriminator_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=self.lr_discriminator, weight_decay=self.decay)
        critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=self.lr_critic, weight_decay=self.decay)

        update_ema(self.ema, self.model, 0)

        metrics = {}

        update_times = 0
        batch_iter = iter(self.dataloader)
        batch = next(batch_iter, None) 
        if batch is None:
            batch_iter = iter(self.dataloader)
            batch = next(batch_iter)
        batch = batch[0].to(self.device)
        train_generator = False
        train_discriminator = True
        train_critic = True
        history_real_validities = deque(maxlen=10)
        history_fake_validities = deque(maxlen=10)
        cool_down = 0
        FID_list = []
        for epoch in epoch_bar:
            metrics = {}
            self.update_schedule()
            self.set_std_schedule(self.ddpm_start + (self.ddpm_end - self.ddpm_start) * epoch / self.n_epochs)

            self.model.train()
            acc_loss = 0.0

            self.epoch = epoch
            obs = torch.zeros((self.tune_timesteps, self.sample_batch_size, batch.shape[1], batch.shape[2], batch.shape[3]), device=self.device, dtype=torch.float32) # (timesteps, batch_size, channels, height, width)
            timesteps = torch.zeros((self.tune_timesteps, self.sample_batch_size), device=self.device, dtype=torch.long)
            logprobs = torch.zeros((self.tune_timesteps, self.sample_batch_size), device=self.device, dtype=torch.float32)
            actions = torch.zeros((self.tune_timesteps, self.sample_batch_size, batch.shape[1], batch.shape[2], batch.shape[3]), device=self.device, dtype=torch.float32) # (timesteps, batch_size, channels, height, width)
            values = torch.zeros((self.tune_timesteps, self.sample_batch_size), device=self.device, dtype=torch.float32) # (timesteps, batch_size)
            rewards = torch.zeros((self.tune_timesteps, self.sample_batch_size), device=self.device, dtype=torch.float32) # (timesteps, batch_size)
            next_obs = torch.randn((self.sample_batch_size, batch.shape[1], batch.shape[2], batch.shape[3]), device=self.device, dtype=torch.float32)
            fake_validities = torch.zeros((self.tune_timesteps, self.sample_batch_size), device=self.device, dtype=torch.float32)
            action_mu = torch.zeros((self.tune_timesteps, self.sample_batch_size), device=self.device, dtype=torch.float32)
            action_sigma = torch.zeros((self.tune_timesteps, self.sample_batch_size), device=self.device, dtype=torch.float32)
            with torch.no_grad():
                for i in tqdm(range(self.timesteps - 1, self.tune_timesteps - 1, -self.timesteps // self.ref_sample_timesteps), desc='Rolling image trajectory with ref model', leave=False):
                    next_obs = self.ref_rollout.p_sample(self.model, next_obs, torch.full((self.sample_batch_size,), i, device=self.device), i)
                total_reward = torch.zeros((self.sample_batch_size,), device=self.device, dtype=torch.float32)
                for i in tqdm(range(self.tune_timesteps - 1, -1, -1), desc='Rolling image trajectory with finetuned model', leave=False):
                    obs[self.tune_timesteps - i - 1] = next_obs
                    timesteps[self.tune_timesteps - i - 1] = torch.full((self.sample_batch_size,), i, device=self.device)
                    action = self.model(next_obs, timesteps[self.tune_timesteps - i - 1])
                    action_mu[self.tune_timesteps - i - 1] = (np.sqrt(1 - self.std ** 2) * self.k_ddpm[i]).to(self.device)
                    action_sigma[self.tune_timesteps - i - 1] = (torch.sqrt(self.std ** 2 * self.k_ddpm[i] ** 2 + self.std_ddpm[i] ** 2)).to(self.device)
                    dist = torch.distributions.Normal(action_mu[self.tune_timesteps - i - 1].reshape(-1, 1, 1, 1) * action, action_sigma[self.tune_timesteps - i - 1].reshape(-1, 1, 1, 1))
                    action = dist.sample()
                    logprobs[self.tune_timesteps - i - 1] = dist.log_prob(action).mean(dim=(1, 2, 3))
                    values[self.tune_timesteps - i - 1] = self.critic(next_obs, timesteps[self.tune_timesteps - i - 1]).reshape(-1)

                    actions[self.tune_timesteps - i - 1] = action
                    next_obs = self.rollout.p_action_sample(action, next_obs, timesteps[self.tune_timesteps - i - 1], i)
                    fake_validities[self.tune_timesteps - i - 1] = self.discriminator(next_obs, torch.full((self.sample_batch_size,), i - 1, device=self.device)).view(-1)
                    rewards[self.tune_timesteps - i - 1] = (fake_validities[self.tune_timesteps - i - 1] * (self.reward_max - self.reward_min) + self.reward_min) * self.reward_schedule[i]
                    total_reward += rewards[self.tune_timesteps - i - 1]

            metrics.update({
                'total_reward': total_reward.mean().item(),
                'last_fake_validity': self.discriminator(next_obs, torch.full((self.sample_batch_size,), -1, device=self.device)).mean().item(),
                'tune_timesteps': self.tune_timesteps,
            })

            advantages, returns = self.calculate_advantage(values, rewards)

            # log sample images
            with torch.no_grad():
                plt.figure(figsize=(10, 10))
                image = next_obs.cpu() * 0.5 + 0.5
                image = image.clamp(0, 1)
                grid = make_grid(image, nrow=int(np.sqrt(image.shape[0])), normalize=False, padding=0)
                plt.imshow(grid.permute(1, 2, 0))
                plt.xticks([])
                plt.yticks([])
                
                metrics["Image/train"] = wandb.Image(plt)
                plt.close()

            plot_curve(
                x_vals=list(reversed(range(self.tune_timesteps))),
                y_vals=values.mean(dim=1),
                x_label='Timesteps',
                y_label='Value',
                x_min=0,
                x_max=self.timesteps
            )
            metrics['Curve/Value'] = wandb.Image(plt)
            plt.close()

            plot_curve(
                x_vals=list(reversed(range(self.tune_timesteps))),
                y_vals=advantages.mean(dim=1),
                x_label='Timesteps',
                y_label='Advantage',
                x_min=0,
                x_max=self.timesteps
            )
            metrics['Curve/Advantage'] = wandb.Image(plt)
            plt.close()

            plot_curve(
                x_vals=list(reversed(range(self.tune_timesteps))),
                y_vals=returns.mean(dim=1),
                x_label='Timesteps',
                y_label='Return',
                x_min=0,
                x_max=self.timesteps
            )
            metrics['Curve/Return'] = wandb.Image(plt)
            plt.close()

            plot_curve(
                x_vals=list(reversed(range(self.tune_timesteps))), 
                y_vals=rewards.mean(dim=1), 
                x_label='Timesteps', 
                y_label='Reward',
                y_min=-1,
                y_max=1,
                x_min=0,
                x_max=self.timesteps
            )
            metrics['Curve/Reward'] = wandb.Image(plt)
            plt.close()
            plot_curve(
                x_vals=list(reversed(range(self.tune_timesteps))), 
                y_vals=fake_validities.mean(dim=1), 
                x_label='Timesteps', 
                y_label='Fake Validity',
                y_min=0,
                y_max=1,
                x_min=0,
                x_max=self.timesteps
            )
            metrics['Curve/Fake Validity'] = wandb.Image(plt)
            plt.close()
            plot_curve(
                x_vals=range(self.tune_timesteps), 
                y_vals=self.reward_schedule[:self.tune_timesteps],
                x_label='Timesteps',
                y_label='Reward Schedule',
                y_min=0,
                y_max=1,
                x_min=0,
                x_max=self.timesteps
            )
            metrics['Curve/Reward Schedule'] = wandb.Image(plt)
            plt.close()
            plot_curve(
                x_vals=range(self.tune_timesteps), 
                y_vals=self.discriminator_schedule[:self.tune_timesteps].cpu().numpy(),
                x_label='Timesteps',
                y_label='Discriminator Schedule',
                y_min=0,
                y_max=1,
                x_min=0,
                x_max=self.timesteps
            )
            metrics['Curve/Discriminator Schedule'] = wandb.Image(plt)
            plt.close()
            plot_curve(
                x_vals=range(self.tune_timesteps), 
                y_vals=action_mu[:self.tune_timesteps][:, 0].cpu().flip(0).numpy(),
                x_label='Timesteps',
                y_label='Action Mu',
                x_min=0,
                x_max=self.timesteps
            )
            metrics['Curve/Action Mu'] = wandb.Image(plt)
            plt.close()
            plot_curve(
                x_vals=range(self.tune_timesteps), 
                y_vals=action_sigma[:self.tune_timesteps][:, 0].cpu().flip(0).numpy(),
                x_label='Timesteps',
                y_label='Action Sigma',
                x_min=0,
                x_max=self.timesteps
            )
            metrics['Curve/Action Sigma'] = wandb.Image(plt)
            plt.close()

            # plot all batch returns
            plt.figure(figsize=(10, 6))
            for i in range(self.sample_batch_size):
                plt.plot(returns.cpu().numpy()[:, i].flatten(), label=f'Batch {i}')
            metrics['Full_Curve/Returns'] = wandb.Image(plt)
            plt.close()

            # plot all batch values
            plt.figure(figsize=(10, 6))
            for i in range(self.sample_batch_size):
                plt.plot(values.cpu().numpy()[:, i].flatten(), label=f'Batch {i}')
            metrics['Full_Curve/Values'] = wandb.Image(plt)
            plt.close()


            if epoch < self.n_epochs_train_only_value:
                train_generator = False
                train_discriminator = True
                train_critic = True
            elif np.mean(history_fake_validities) < np.mean(history_real_validities) - self.real_fake_gap:
                cool_down = 20
                train_generator = True
                train_discriminator = False
                train_critic = True
            elif np.mean(history_fake_validities) > np.mean(history_real_validities) - self.real_fake_gap:
                train_generator = True
                train_discriminator = True
                train_critic = True

            metrics.update({
                'smoothed_fake_validity': np.mean(history_fake_validities),
                'smoothed_real_validity': np.mean(history_real_validities),
            })

            with torch.no_grad():
                b_obs = obs.reshape(-1, *obs.shape[2:])
                b_actions = actions.reshape(-1, *actions.shape[2:])
                b_returns = returns.reshape(-1)
                b_advantages = advantages.reshape(-1)
                b_timesteps = timesteps.reshape(-1)
                b_values = values.reshape(-1)
                b_logprobs = logprobs.reshape(-1)
                b_mu = action_mu.reshape(-1)
                b_sigma = action_sigma.reshape(-1)
            
            # Train
            b_size = b_obs.size(0)
            b_inds = np.arange(b_size)
            np.random.shuffle(b_inds)

            # Train discriminator
            # weighted_fake_validity = (rewards.mean(dim=1).sum() / self.total_reward_weight - self.reward_min) / (self.reward_max - self.reward_min)
            # metrics['weighted_fake_validity'] = weighted_fake_validity
            real_validity_total = 0
            fake_validity_total = 0
            d_loss_total = 0
            d_grad_total = 0
            cnt = 0
            for start in tqdm(range(0, b_size - self.batch_size + 1, self.batch_size), desc='Discriminator Update', leave=False):
                batch = next(batch_iter, None) 
                if batch is None:
                    batch_iter = iter(self.dataloader)
                    batch = next(batch_iter)
                batch = batch[0].to(self.device)
                while batch.shape[0] < self.batch_size:
                    batch = next(batch_iter, None) 
                    if batch is None:
                        batch_iter = iter(self.dataloader)
                        batch = next(batch_iter)
                    batch = batch[0].to(self.device)
                with torch.no_grad():
                    real_imgs = batch
                    if self.vae is not None:
                        real_imgs = self.vae.encode(real_imgs).latent_dist.sample().mul_(0.18215)
                    end = start + self.batch_size
                    batch_inds = b_inds[start:end]
                    batch_obs = b_obs[batch_inds]
                    batch_timesteps = b_timesteps[batch_inds]
                    batch_gt_obs = self.forward_diffusion_model.q_sample(x_start=real_imgs, t=batch_timesteps)
                
                fake_validity = self.discriminator(batch_obs, batch_timesteps).view(-1, 1)
                real_validity = self.discriminator(batch_gt_obs, batch_timesteps).view(-1, 1)

                d_loss_weights = extract_time_index(self.discriminator_schedule, batch_timesteps, torch.Size([self.batch_size, 1]))

                d_loss = adversarial_loss(fake_validity, fake) * d_loss_weights + adversarial_loss(real_validity, valid) * d_loss_weights
                d_loss = d_loss.mean(dim=(0, 1))

                if train_discriminator:
                    discriminator_optimizer.zero_grad()
                    d_loss.backward()
                    d_grad_total += torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 2000)
                    discriminator_optimizer.step()

                real_validity_total += (real_validity * d_loss_weights).sum().item()
                fake_validity_total += (fake_validity * d_loss_weights).sum().item()
                d_loss_total += d_loss.item()
                cnt += 1

            batch = next(batch_iter, None) 
            if batch is None:
                batch_iter = iter(self.dataloader)
                batch = next(batch_iter)
            batch = batch[0].to(self.device)
            while batch.shape[0] < self.batch_size:
                batch = next(batch_iter, None) 
                if batch is None:
                    batch_iter = iter(self.dataloader)
                    batch = next(batch_iter)
                batch = batch[0].to(self.device)
            batch_gt_obs = batch[:next_obs.shape[0]]
            batch_timesteps = torch.full((next_obs.shape[0],), -1, device=self.device)
            real_validity = self.discriminator(batch_gt_obs, batch_timesteps).view(-1, 1)
            fake_validity = self.discriminator(next_obs, batch_timesteps).view(-1, 1)
            d_loss = adversarial_loss(fake_validity, torch.zeros_like(fake_validity)) + adversarial_loss(real_validity, torch.ones_like(real_validity))
            d_loss = d_loss.mean(dim=(0, 1))
            if train_discriminator:
                discriminator_optimizer.zero_grad()
                d_loss.backward()
                d_grad_total += torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), float('inf'))
                discriminator_optimizer.step()

            real_validity_total += real_validity.sum().item()
            fake_validity_total += fake_validity.sum().item()
            d_loss_total += d_loss.item()
            cnt += 1

            metrics.update({
                'loss/discriminator_loss': d_loss_total / cnt,
                'real_validity': real_validity_total / self.sample_batch_size / (self.total_discriminator_weight + 1),
                'fake_validity': fake_validity_total / self.sample_batch_size / (self.total_discriminator_weight + 1),
                'reward': rewards.mean(dim=0).mean(),
                'grad/discriminator': d_grad_total / cnt,
                # 'gradient_penalty': gradient_penalty.item(),
            })
            history_fake_validities.append(metrics['fake_validity'])
            history_real_validities.append(metrics['real_validity'])

            b_size = b_obs.size(0)

            policy_loss_total = 0
            v_loss_total = 0
            kl_loss_total = 0
            entropy_loss_total = 0
            approx_kl = torch.tensor(0.0, device=self.device)
            old_approx_kl = torch.tensor(0.0, device=self.device)
            actor_grad_total = 0
            critic_grad_total = 0
            cnt = 0

            for start in tqdm(range(0, b_size - self.batch_size + 1, self.batch_size), desc='Model Update', leave=False):
                with torch.no_grad():
                    end = start + self.batch_size
                    batch_inds = b_inds[start:end]
                    batch_obs = b_obs[batch_inds]
                    batch_actions = b_actions[batch_inds]
                    batch_returns = b_returns[batch_inds]
                    batch_advantages = b_advantages[batch_inds]
                    batch_timesteps = b_timesteps[batch_inds]
                    batch_values = b_values[batch_inds]
                    batch_logprobs = b_logprobs[batch_inds]
                    batch_mu = b_mu[batch_inds]
                    batch_sigma = b_sigma[batch_inds]

                action_mean = self.model(batch_obs, batch_timesteps)
                dist = torch.distributions.Normal(batch_mu.reshape(-1, 1, 1, 1) * action_mean, batch_sigma.reshape(-1, 1, 1, 1))
                new_logprobs = dist.log_prob(batch_actions).mean(dim=(1, 2, 3))
                entropy = dist.entropy().mean(dim=(1, 2, 3))
                new_value = self.critic(batch_obs, batch_timesteps)
                logratio = new_logprobs - batch_logprobs
                ratio = logratio.exp()
                policy_loss = torch.tensor(0.0, device=self.device)

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    ref_action = self.model.ref_model(batch_obs, batch_timesteps)
                kl_loss = F.mse_loss(action_mean, ref_action)
                policy_loss += kl_loss * self.kl_coef


                if self.norm_adv:
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

                pg_loss1 = -ratio * batch_advantages
                pg_loss2 = -torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * batch_advantages
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                new_value = new_value.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (new_value - batch_returns) ** 2
                    v_clipped = batch_values + torch.clamp(new_value - batch_values, -self.clip_coef * torch.abs(batch_values), self.clip_coef * torch.abs(batch_values))
                    v_loss_clipped = (v_clipped - batch_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * (new_value - batch_returns) ** 2
                    v_loss = v_loss.mean()

                # if v_loss > 200:
                #     train_generator = False

                entropy_loss = -entropy.mean() * self.entropy_coef

                ppo_loss = pg_loss + entropy_loss

                policy_loss += ppo_loss
                loss = (
                    (policy_loss if train_generator else torch.tensor(0.0, device=self.device))
                    +
                    (v_loss * self.vf_coef if train_critic else torch.tensor(0.0, device=self.device))
                )

                model_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                loss.backward()
                actor_grad_total += torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
                critic_grad_total += torch.nn.utils.clip_grad_norm_(self.critic.parameters(), float('inf'))
                model_optimizer.step()
                critic_optimizer.step()

                policy_loss_total += policy_loss.item()
                v_loss_total += v_loss.item()
                kl_loss_total += kl_loss.item()
                entropy_loss_total += entropy_loss.item()
                cnt += 1

                if self.target_kl is not None and approx_kl > self.target_kl:
                    print(f"Early stopping at {start}/{b_size} due to reaching max kl {approx_kl:.4f} > {self.target_kl:.4f}")
                    break
            
            if train_generator:
                self.model_update_epoch += 1
            metrics.update({
                'loss/policy_loss': policy_loss_total / cnt,
                'loss/v_loss': v_loss_total / cnt,
                'loss/kl_loss': kl_loss_total / cnt,
                'loss/entropy_loss': entropy_loss_total / cnt,
                'approx_kl': approx_kl.item(),
                'old_approx_kl': old_approx_kl.item(),
                'model_update_epoch': self.model_update_epoch,
                'freeze/train_generator': int(train_generator),
                'freeze/train_discriminator': int(train_discriminator),
                'freeze/train_critic': int(train_critic),
                'grad/model': actor_grad_total / cnt,
                'grad/critic': critic_grad_total / cnt,
                'lr/model': model_optimizer.param_groups[0]['lr'],
                'lr/critic': critic_optimizer.param_groups[0]['lr'],
                'lr/discriminator': discriminator_optimizer.param_groups[0]['lr'],
            })

            update_ema(self.ema, self.model, self.ema_rate)

            # save generated images
            if epoch == 0 or (epoch + 1) % self.sample_and_save_freq == 0:
                samples = self.sampler.sample(model=self.ema, image_size=self.img_size, batch_size=16, channels=self.channels)
                all_images = torch.tensor(samples[-1])

                if self.vae is not None:
                    with torch.no_grad():
                        all_images = self.vae.module.decode(all_images.to(self.device) / 0.18215).sample 
                        all_images = all_images.cpu().detach()

                all_images = all_images * 0.5 + 0.5
                all_images = all_images.clamp(0, 1)
                fig = plt.figure(figsize=(10, 10))
                grid = make_grid(all_images, nrow=int(np.sqrt(all_images.shape[0])), normalize=False, padding=0)
                plt.imshow(grid.permute(1, 2, 0))
                plt.xticks([])
                plt.yticks([])
                
                #save figure wandb
                metrics["Image/Samples"] = wandb.Image(fig)
                plt.close(fig)

            if acc_loss/len(self.dataloader.dataset) < best_loss:
                best_loss = acc_loss/len(self.dataloader.dataset)

            if epoch % self.snapshot == 0:
                torch.save(self.ema.state_dict(), self.models_dir / f"ema_{'Lat_' if self.vae is not None else ''}{self.dataset}_epoch{epoch}.pt")
                torch.save(self.model.state_dict(), self.models_dir / f"model_{'Lat_' if self.vae is not None else ''}{self.dataset}_epoch{epoch}.pt")
                torch.save(self.discriminator.state_dict(), self.models_dir / f"discriminator_{'Lat_' if self.vae is not None else ''}{self.dataset}_epoch{epoch}.pt")

            if self.eval_fid_every > 0 and self.model_update_epoch % self.eval_fid_every == 0 and self.model_update_epoch not in FID_list:
                FID_list.append(self.model_update_epoch)
                fid_score = self.eval_fid()
                metrics[f'FID/FID_num={self.eval_fid_num}_step={self.eval_fid_steps}'] = fid_score
                ema_fid_score = self.eval_ema_fid()
                metrics[f'FID/ema_FID_num={self.eval_fid_num}_step={self.eval_fid_steps}'] = ema_fid_score

            metrics["epoch"] = epoch
            wandb.log(metrics)