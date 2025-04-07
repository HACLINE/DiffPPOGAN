import torch
import torch.nn as nn

from src.models.utils.encoder import TimestepEmbedEncoder

class TimestepEmbedDiscriminator(TimestepEmbedEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_layer = nn.Sequential(
            nn.Linear(self.output_size, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x, timesteps):
        x = super().forward(x, timesteps)
        x = self.final_layer(x)
        return x