GENERATED_IMAGES_DIR=$1
GPU_ID=0

python -m src.eval.FID \
    --gpu_id $GPU_ID \
    --real_images_dir /home/xqc/DiffPPOGAN/data/real/cifar10/imgs \
    --generated_images_dir $GENERATED_IMAGES_DIR