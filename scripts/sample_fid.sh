export DATA_DIR=/home/xqc/DiffPPOGAN/data

python -m src.eval.sample_FID \
    gpu_id=4 \
    fid.real_image_path=$DATA_DIR/real/cifar10/imgs \
    ref_model_path=/home/xqc/DiffPPOGAN/models/ref/cifar.pt \
    checkpoint=/home/xqc/DiffPPOGAN/models/2025-04-07_15-41-31/model_cifar10_epoch400.pt \
    +fid_5k=true \
