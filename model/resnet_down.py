
import torch
import torch.nn as nn
from diffusers.models.resnet import ResnetBlock2D, Downsample2D
from diffusers.models.embeddings import Timesteps
    
class PoseEncoder(nn.Module):
    def __init__(self, vae_chanel=8, in_channels=320, channels=[320, 640, 1280]):
        super().__init__()
        self.conv_in = nn.Conv2d(vae_chanel, in_channels, kernel_size=1)
        resnets = []
        downsamplers = []
        zero_conv = []
        for i in range(len(channels)):
            in_channels = in_channels if i == 0 else channels[i - 1]
            out_channels = channels[i]
            resnets.append(ResnetBlock2D(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=None, # no time embed
            ))
            now_zero_conv = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            )
            nn.init.zeros_(now_zero_conv.weight)
            nn.init.zeros_(now_zero_conv.bias)
            zero_conv.append(now_zero_conv)
            downsamplers.append(Downsample2D(
                out_channels,
                use_conv=False,
                out_channels=out_channels,
                padding=1,
                name="op"
            ) if i != len(channels) - 1 else nn.Identity())

        self.resnets = nn.ModuleList(resnets)
        self.downsamplers = nn.ModuleList(downsamplers)
        self.zero_convs = nn.ModuleList(zero_conv)

    def forward(self, hidden_states):
        features = []
        hidden_states = self.conv_in(hidden_states)
        for resnet, downsampler, zero_conv in zip(self.resnets, self.downsamplers, self.zero_convs):
            hidden_states = resnet(hidden_states, temb=None)
            hidden_states = zero_conv(hidden_states)
            features.append(hidden_states)
            hidden_states = downsampler(hidden_states)
        return features
    
    
# model = PoseEncoder()
# test_input = torch.rand([5, 8, 32, 32])
# test_out = model(test_input)
# for feature in test_out:
#     print(feature.shape)
#     print(feature.mean(), feature.var())

