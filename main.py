from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

first_diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           
    sampling_timesteps = 250   
)

trainer = Trainer(
    first_diffusion,
    'images',
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 700000,         
    gradient_accumulate_every = 2,    
    ema_decay = 0.995,                
    amp = True,                       
    calculate_fid = True             
)

model2 = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

second_diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           
    sampling_timesteps = 250    
)

trainer2 = Trainer(
    first_diffusion,
    './results',
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 700000,        
    gradient_accumulate_every = 2,   
    ema_decay = 0.995,               
    amp = True,                       
    calculate_fid = True              
)

trainer2.train()