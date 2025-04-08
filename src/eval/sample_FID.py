import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval)

@hydra.main(config_path="../config", config_name="sample")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    single_cfg = OmegaConf.create({"_target_": cfg._target_})
    workspace = hydra.utils.instantiate(single_cfg)
    workspace.setup(cfg.cfg)
    if cfg.fid_5k:
        workspace.fid_5k_sample(batch_size=500)
    else:
        workspace.fid_sample(batch_size=500)

if __name__ == "__main__":
    main()