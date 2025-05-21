export DATA_DIR=$PWD/data

python -m src.eval.sample_FID \
    cfg.gpu_id=0 \
    cfg.fid.real_image_path=$DATA_DIR/real/cifar10/imgs \
    cfg.ref_model_path=pretrained/base.pt \
    cfg.checkpoint=pretrained/best.pt