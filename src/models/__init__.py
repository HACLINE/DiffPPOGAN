import hydra
from omegaconf import DictConfig, OmegaConf

def get_model(cfg: DictConfig):
    if type(cfg) == dict:
        cfg = OmegaConf.create(cfg)
    return hydra.utils.instantiate(cfg)