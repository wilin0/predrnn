__author__ = 'yunbo'

import torch
import torch.nn as nn

class Gradient_Highway_Unit(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(Gradient_Highway_Unit, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 2, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 2, width, width])
            )
            self.conv_z = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 2, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 2, width, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 2, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_z = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 2, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )

    def forward(self, x_t, z_t):
        x_concat = self.conv_x(x_t)
        z_concat = self.conv_z(z_t)
        p_x, s_x = torch.split(x_concat, self.num_hidden, dim=1)
        p_z, s_z = torch.split(z_concat, self.num_hidden, dim=1)

        p_t = torch.tanh(p_x + p_z)
        s_t = torch.sigmoid(s_x + s_z)

        z_new = s_t * p_t + (1 - s_t) * z_t

        return z_new
