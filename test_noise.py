import torch
import numpy as np
from diffusers import DDPMScheduler

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
pred_noise = torch.rand(8, 3, 32, 32)
sample = torch.randn(8, 3, 32, 32)
timesteps = torch.randint(1, 1000, (8,) )

tt = torch.from_numpy(np.array(range(1, 9)))
tt = tt.view(tt.shape[0], 1, 1, 1)


### 定义获取x0的函数
def pred_x0_from_noise(noise_scheduler, sample, pred_noise, timesteps):
    t = timesteps
    alpha_prod_t = noise_scheduler.alphas_cumprod[t]
    alpha_prod_t_prev = noise_scheduler.alphas_cumprod[torch.clamp(t - 1, min=0)]
    beta_prod_t = (1 - alpha_prod_t).view(timesteps.shape[0], 1, 1, 1)
    beta_prod_t_prev = (1 - alpha_prod_t_prev).view(timesteps.shape[0], 1, 1, 1)

    pred_original_sample = (sample - (beta_prod_t ** (0.5)) * pred_noise) / (alpha_prod_t.view(timesteps.shape[0], 1, 1, 1) ** (0.5))
    pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
    return pred_original_sample

print(sample[1])
sample1 = sample / tt
print(sample1[1])


# pred_original_sample = pred_x0_from_noise(noise_scheduler, sample, pred_noise, timesteps)
# print(pred_original_sample.shape)
# Update sample with step
# sample = noise_scheduler.step(residual, t, sample).prev_sample

# show_images(sample)

