import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval)

@hydra.main(config_path="../config", config_name="train")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    single_cfg = OmegaConf.create({"_target_": cfg._target_})
    workspace = hydra.utils.instantiate(single_cfg)
    workspace.setup(cfg.cfg)
    workspace.sample()

if __name__ == "__main__":
    main()