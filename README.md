# Enhancing Pre-trained Diffusion Models with Reinforcement Learning and Adversarial Reward Functions

## Contributors
- [Qicheng Xu](https://github.com/HACLINE)
- [Zehua Wang](https://github.com/patrickwzh)

## Setup
```bash
conda env create -f environment.yaml
conda activate diffppogan
```
Dataset will be automatically downloaded when running the code. You may refer to [Generative Zoo](https://github.com/caetas/GenerativeZoo) for more details on the dataset.

## Training
We didn't provide the training script for the standard diffusion model. You can use the codes in [Generative Zoo](https://github.com/caetas/GenerativeZoo) to train the standard diffusion model, or you can use the pre-trained model provided in `pretrained/base.pt`. 

Please refer to `scripts/train.sh` for the training script. You can run the following command to start training:
```bash
export DATA_DIR=$PWD/data

python -m src.train.train \
    cfg=adv_schedule_r3 \
    cfg.gpu_id=0 \
    cfg.fid.real_image_path=$DATA_DIR/real/cifar10/imgs \
    cfg.ref_model_path=pretrained/base.pt \
    cfg.wandb.name=WANDB_NAME \
```

## Evaluation
You can sample images from the pre-trained model using `scripts/sample.sh` the following command:
```bash
export DATA_DIR=$PWD/data

python -m src.sample.sample \
    cfg=adv_schedule_r3 \
    cfg.gpu_id=0 \
    cfg.ref_model_path=pretrained/base.pt \
    cfg.fid.real_image_path=DATA_DIR/real/cifar10/imgs \
    cfg.checkpoint=pretrained/best.pt
```

Run `scripts/sample_fid.sh` to sample images and use `scripts/eval_fid.sh` to evaluate the FID score. 