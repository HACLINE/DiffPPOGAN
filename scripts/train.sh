export DATA_DIR=/home/wzh/nfsdata/projects/DiffPPOGAN/data

python -m src.train.train \
    cfg.gpu_id=5 \
    cfg.fid.real_image_path=$DATA_DIR/real/cifar10/imgs \
    cfg.ref_model_path=/home/xqc/DiffPPOGAN/models/ref/cifar.pt \
    cfg.wandb.name=better_adv_lr1e-4 \
    cfg.lr=1e-4 \
    cfg.n_epochs=10000