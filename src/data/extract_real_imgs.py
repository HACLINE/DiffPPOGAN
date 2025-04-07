import torch
import os
from src.data.Dataloaders import *
import argparse
from pathlib import Path
import cv2

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str)
    argparser.add_argument('--gpu_id', type=int, default=0)
    args = argparser.parse_args()

    real_images_dir = f"{os.environ.get('DATA_DIR')}/real/{args.dataset}/imgs"
    if not os.path.exists(real_images_dir):
        os.makedirs(real_images_dir)
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    dataloaders = []
    for split in ['train', 'val']:
        dataloaders.append(pick_dataset(args.dataset, split, 1, normalize=True, size=None, num_workers=1)[0])

    cnt = 0
    for dataloader in dataloaders:
        for images, _ in dataloader:
            samps = images.cpu().numpy()
            samps = samps * 0.5 + 0.5
            samps = samps.clip(0, 1)
            samps = samps.transpose(0,2,3,1)
            samps = (samps * 255).astype(np.uint8)
            for samp in samps:
                cv2.imwrite(f"{real_images_dir}/{cnt}.png", cv2.cvtColor(samp, cv2.COLOR_RGB2BGR) if samp.shape[-1] == 3 else samp)
                cnt += 1
