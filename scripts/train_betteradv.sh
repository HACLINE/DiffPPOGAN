export DATA_DIR=/home/wzh/nfsdata/projects/DiffPPOGAN/data

python -m src.train.train \
    cfg.gpu_id=7 \
    cfg.fid.real_image_path=$DATA_DIR/real/cifar10/imgs \
    cfg.ref_model_path=/home/wzh/tmp/opencv/cifar.pt \
    cfg.wandb.name=better_adv \
    cfg.lr=1e-4 \
    cfg.n_epochs=10000