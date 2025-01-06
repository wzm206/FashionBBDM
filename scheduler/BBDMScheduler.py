import numpy as np
import torch

class BBDMScheduler(): 
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        max_var=1.0, 
    ):
        self.num_train_timesteps = num_train_timesteps
        self.max_var = max_var
        
        T = self.num_train_timesteps
        self.mt_type = "linear"
        
        if self.mt_type == "linear":
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T)
        elif self.mt_type == "sin":
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
        else:
            raise NotImplementedError
        m_tminus = np.append(0, m_t[:-1])

        variance_t = 2. * (m_t - m_t ** 2) * self.max_var
        variance_tminus = np.append(0., variance_t[:-1])
        variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t

        self.m_t = torch.from_numpy(m_t.astype(np.float32))
        self.m_tminus = torch.from_numpy(m_tminus.astype(np.float32))
        self.variance_t = torch.from_numpy(variance_t.astype(np.float32))
        self.variance_tminus = torch.from_numpy(variance_tminus.astype(np.float32))
        self.posterior_variance_t = torch.from_numpy(posterior_variance_t.astype(np.float32))
        
    def add_noise(
        self,
        original_samples: torch.Tensor,
        target_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        timesteps = timesteps.to(original_samples.device)
        m_t = self.m_t.to(original_samples.device)[timesteps]
        var_t = self.variance_t.to(original_samples.device)[timesteps]
        sigma_t = torch.sqrt(var_t)
        
        while len(sigma_t.shape) < len(original_samples.shape):
            sigma_t = sigma_t.unsqueeze(-1)
            m_t = m_t.unsqueeze(-1)
            
        # if self.objective == 'grad':
        #     objective = m_t * (y - x0) + sigma_t * noise
        # elif self.objective == 'noise':
        #     objective = noise
        # elif self.objective == 'ysubx':
        #     objective = y - x0
        # else:
        #     raise NotImplementedError()

        return (1. - m_t) * original_samples + m_t * target_samples + sigma_t * noise

        # if self.skip_sample:
        #     if self.sample_type == 'linear':
        #         midsteps = torch.arange(self.num_timesteps - 1, 1,
        #                                 step=-((self.num_timesteps - 1) / (self.sample_step - 2))).long()
        #         self.steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)
        #     elif self.sample_type == 'cosine':
        #         steps = np.linspace(start=0, stop=self.num_timesteps, num=self.sample_step + 1)
        #         steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.) / 2. * self.num_timesteps
        #         self.steps = torch.from_numpy(steps)
        # else:
        #     self.steps = torch.arange(self.num_timesteps-1, -1, -1)
    @torch.no_grad()
    def p_sample(self, y, noisy_residual, i, input, clip_denoised=False,):
        
        self.num_timesteps = 1000
        self.sample_step = 200
        midsteps = torch.arange(self.num_timesteps - 1, 1,
                                step=-((self.num_timesteps - 1) / (self.sample_step - 2))).long()
        self.steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)
        
        t = self.steps[i].to(input.device)
        n_t = self.steps[min(i+1, 199)].to(input.device)
        
        # t = t.to(input.device)
        # nt是小的
        # n_t = max(t-5, 0)
        m_t = self.m_t.to(input.device)[t]
        m_nt = self.m_t.to(input.device)[n_t]
        
        var_t = self.variance_t.to(input.device)[t]
        var_nt = self.variance_t.to(input.device)[n_t]
        sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
        sigma_t = torch.sqrt(sigma2_t)
        sigma_t_pre = torch.sqrt(var_t)
        sigma_nt = torch.sqrt(var_nt)
        
        while len(sigma_t.shape) < len(input.shape):
            sigma_t = sigma_t.unsqueeze(-1)
            m_t = m_t.unsqueeze(-1)
            m_nt = m_nt.unsqueeze(-1)
            sigma2_t = sigma2_t.unsqueeze(-1)
        
        x0_recon = (input - m_t * y - sigma_t_pre * noisy_residual) / (1. - m_t)
        
        # x0_recon = input - noisy_residual
        
        if self.steps[i] == 0:
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            return x0_recon, x0_recon
        else:
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            noise = torch.randn_like(input)
            # 原始方案
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + torch.sqrt((var_nt - sigma2_t) / var_t) * \
                            (input - (1. - m_t) * x0_recon - m_t * y)
            return x_tminus_mean + sigma_t * noise, x0_recon   
            # 无噪声方案             
            # x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y
            # return x_tminus_mean, x0_recon
            
            # 消融实验
            # sigma_wzm = 0.6*sigma_nt
            # k2 = (sigma_nt-sigma_wzm)/sigma_t_pre
            # k1 = 1 - m_nt - (1-m_t)*k2
            # k3 = m_nt - m_t*k2
            # retret = k1*x0_recon + k2*input + k3*y + sigma_wzm*noise
            # return retret, x0_recon
        
    def get_velocity(self, latents, noise, timesteps, latents_condition):
        m_t = self.m_t.to(latents.device)[timesteps]
        var_t = self.variance_t.to(latents.device)[timesteps]
        sigma_t = torch.sqrt(var_t)
        while len(sigma_t.shape) < len(latents.shape):
            sigma_t = sigma_t.unsqueeze(-1)
            m_t = m_t.unsqueeze(-1)

        # if self.objective == 'grad':
        objective = m_t * (latents_condition - latents) + sigma_t * noise
        return objective
        
        
# Scheduler = BBDMScheduler()

# latents = torch.rand([2, 1, 8, 8])
# target = torch.randn_like(latents)
# noise = torch.randn_like(latents)
# timesteps = torch.tensor([0, 999]).long()
# noisy_latents = Scheduler.add_noise(latents, target, noise, timesteps)
# print("latents", latents)
# print("target", target)
# print("out", noisy_latents)
          
