from src.workspace.diffppogan import DiffPPOGANWorkspace

import hydra
import torch
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval)

@hydra.main(config_path="../config", config_name="base")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    device = f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() and cfg.gpu_id >= 0 else "cpu"
    workspace = DiffPPOGANWorkspace(cfg, device)
    workspace.train_model()

if __name__ == "__main__":
    main()