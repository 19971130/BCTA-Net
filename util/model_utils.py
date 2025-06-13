import torch
import torch.nn as nn
from model.component.norm import LayerNorm2d


def get_norm_func(func_name: str, channels):
    func_name = func_name.upper()
    if func_name == 'BATCHNORM':
        return nn.BatchNorm2d(channels)
    elif func_name == 'LAYERNORM':
        return LayerNorm2d(channels)
    elif func_name == 'INSTANCENORM':
        return nn.InstanceNorm2d(channels)
    else:
        raise RuntimeError('Failure to match function: {}'.format(func_name))


def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x
