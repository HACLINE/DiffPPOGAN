import torch
import torch.nn as nn
from copy import deepcopy

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

class FineTuningModel(BaseModel):
    def __init__(self, tune_timesteps):
        super(FineTuningModel, self).__init__()
        self.ref_model = None
        self.tune_timesteps = tune_timesteps

    def store_ref_model(self, ref_model):
        self.ref_model = deepcopy(ref_model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def filter_forward_output(self, finetuned_output, x, timesteps, y=None):
        """
        Filter the output of the model using the reference model.
        """
        if self.ref_model is None:
            return finetuned_output
        
        if torch.all(timesteps < self.tune_timesteps):
            return finetuned_output

        mask = timesteps > self.tune_timesteps - 1
        with torch.no_grad():
            ref_output = self.ref_model(x, timesteps)
        
        return finetuned_output * ~mask[:, None, None, None] + ref_output * mask[:, None, None, None]
        