export DATA_DIR=/home/xqc/DiffPPOGAN/data

python -m src.train.train \
    gpu_id=2 \
    fid.real_image_path=$DATA_DIR/real/cifar10/imgs \
    ref_model_path=/home/xqc/DiffPPOGAN/models/ref/DDPM_cifar10_epoch200.pt \
    wandb.name=base