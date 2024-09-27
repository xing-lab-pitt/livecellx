import argparse
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from pathlib import Path
from datetime import datetime
import torch

# Create the parser
parser = argparse.ArgumentParser(description="Train a diffusion model")

# Add the arguments
parser.add_argument("--sc_img_folder", type=str, required=True, help="The path to the sc image folder")
parser.add_argument("--results_folder", type=str, required=True, help="The path to the results folder")
parser.add_argument("--train_num_steps", type=int, default=100000, help="The number of training steps")
parser.add_argument("--image_size", type=int, default=128, help="The image size")
parser.add_argument("--batch_size", type=int, default=8, help="The batch size")
parser.add_argument("--model_ckpt_path", type=str, required=True, help="The path to the model checkpoint")

# Parse the arguments
args = parser.parse_args()

sc_img_folder = Path(args.sc_img_folder)
results_folder = str(Path(args.results_folder))


model = Unet(dim=64, dim_mults=(1, 2, 4, 8))

diffusion = GaussianDiffusion(
    model,
    image_size=args.image_size,  # size of image
    timesteps=1000,  # number of steps
    sampling_timesteps=250,  # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    # loss_type = 'l1'            # L1 or L2
)

# # Load mode pth
if args.model_ckpt_path is not None:
    diffusion.load_state_dict(torch.load(args.model_ckpt_path))

trainer = Trainer(
    diffusion,
    str(sc_img_folder),
    train_batch_size=args.batch_size,
    train_lr=8e-5,
    train_num_steps=args.train_num_steps,
    gradient_accumulate_every=2,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    # amp = True,                       # turn on mixed precision
    calculate_fid=True,  # whether to calculate fid during training
    results_folder=results_folder,
)

# trainer.load(milestone=1)
trainer.train()
