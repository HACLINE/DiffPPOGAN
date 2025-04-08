import os
from pytorch_fid import fid_score
import torch
import argparse
from pathlib import Path
import torch

parser = argparse.ArgumentParser(description='FID')
parser.add_argument('--generated_images_dir', '-g', type=str)
parser.add_argument('--real_images_dir', '-r', type=str)
args = parser.parse_args()

generated_images_dir = args.generated_images_dir
real_images_dir = args.real_images_dir

with torch.no_grad():
    block_idx = fid_score.InceptionV3.BLOCK_INDEX_BY_DIM[2048]

    model = fid_score.InceptionV3([block_idx]).to('cuda')

    if Path(real_images_dir).is_dir():
        m2, s2 = fid_score.compute_statistics_of_path(real_images_dir, model, 500, 2048, 'cuda', 8)
    else:
        m2, s2 = torch.load(real_images_dir, weights_only=False)

    if Path(generated_images_dir).is_dir():
        m1, s1 = fid_score.compute_statistics_of_path(generated_images_dir, model, 500, 2048, 'cuda', 8)
    else:
        m1, s1 = torch.load(generated_images_dir, weights_only=False)
    
    fid_value = fid_score.calculate_frechet_distance(m1, s1, m2, s2)

print(f'FID: {fid_value}')