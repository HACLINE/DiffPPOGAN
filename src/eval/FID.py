import os
from pytorch_fid import fid_score
import torch
import argparse

parser = argparse.ArgumentParser(description='FID')
parser.add_argument('--generated_images_dir', '-g', type=str)
parser.add_argument('--real_images_dir', '-r', type=str)
args = parser.parse_args()

generated_images_dir = args.generated_images_dir
real_images_dir = args.real_images_dir

with torch.no_grad():
    fid_value = fid_score.calculate_fid_given_paths([real_images_dir, generated_images_dir], batch_size=64, device='cuda', dims=2048, num_workers=8)
print(f'FID: {fid_value}')