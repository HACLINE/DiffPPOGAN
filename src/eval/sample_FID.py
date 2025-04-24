import hydra
from omegaconf import OmegaConf
from src.eval.FID import eval_fid
import torch

OmegaConf.register_new_resolver("eval", eval)

@hydra.main(config_path="../config", config_name="sample")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    single_cfg = OmegaConf.create({"_target_": cfg._target_})
    workspace = hydra.utils.instantiate(single_cfg)
    workspace.setup(cfg.cfg)
    if cfg.get("fid_5k", False):
        path = workspace.fid_5k_sample(batch_size=500)
    else:
        path = workspace.fid_sample(batch_size=500)
    eval_fid(
        generated_images_dir=path,
        real_images_dir=cfg.cfg.fid.real_image_path,
        device=torch.device(f"cuda:{cfg.cfg.gpu_id}"),
    )

if __name__ == "__main__":
    main()