from src.models.utils.util import *
from src.models.diffusion.base import FineTuningModel
from diffusers.models.unets.unet_2d import UNet2DModel
import torch
import torch.nn as nn

class UNetModel(FineTuningModel, UNet2DModel):
    def __init__(self, 
        sample_size=32,
        in_channels=3,
        out_channels=3,
        center_input_sample=False,
        time_embedding_type="positional",
        time_embedding_dim=None,
        freq_shift=1,
        flip_sin_to_cos=False,
        down_block_types=['DownBlock2D', 'AttnDownBlock2D', 'DownBlock2D', 'DownBlock2D'],
        mid_block_type="UNetMidBlock2D",
        up_block_types=['UpBlock2D', 'UpBlock2D', 'AttnUpBlock2D', 'UpBlock2D'],
        block_out_channels=[128, 256, 256, 256],
        layers_per_block=2,
        mid_block_scale_factor=1,
        downsample_padding=0,
        downsample_type="conv",
        upsample_type="conv",
        dropout=0.0,
        act_fn="silu",
        attention_head_dim=None,
        norm_num_groups=32,
        attn_norm_num_groups=None,
        norm_eps=1e-06,
        resnet_time_scale_shift="default",
        add_attention=True,
        class_embed_type=None,
        num_class_embeds=None,
        num_train_timesteps=None,
        tune_timesteps=0,
    ):
        FineTuningModel.__init__(self, tune_timesteps)
        UNet2DModel.__init__(
            self,
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            center_input_sample=center_input_sample,
            time_embedding_type=time_embedding_type,
            time_embedding_dim=time_embedding_dim,
            freq_shift=freq_shift,
            flip_sin_to_cos=flip_sin_to_cos,
            down_block_types=down_block_types,
            mid_block_type=mid_block_type,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            mid_block_scale_factor=mid_block_scale_factor,
            downsample_padding=downsample_padding,
            downsample_type=downsample_type,
            upsample_type=upsample_type,
            dropout=dropout,
            act_fn=act_fn,
            attention_head_dim=attention_head_dim,
            norm_num_groups=norm_num_groups,
            attn_norm_num_groups=attn_norm_num_groups,
            norm_eps=norm_eps,
            resnet_time_scale_shift=resnet_time_scale_shift,
            add_attention=add_attention,
            class_embed_type=class_embed_type, 
            num_class_embeds=num_class_embeds,
            num_train_timesteps=num_train_timesteps,
        )
    
    def forward(self, sample, timestep):
        return UNet2DModel.forward(self, sample, timestep).sample