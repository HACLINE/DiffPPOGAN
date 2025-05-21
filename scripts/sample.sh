export DATA_DIR=$PWD/data

python -m src.sample.sample \
    cfg=adv_schedule_r3 \
    cfg.gpu_id=0 \
    cfg.ref_model_path=pretrained/base.pt \
    cfg.fid.real_image_path=DATA_DIR/real/cifar10/imgs \
    cfg.checkpoint=pretrained/best.pt