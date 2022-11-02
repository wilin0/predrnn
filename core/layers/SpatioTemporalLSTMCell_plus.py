__author__ = 'willin'

import torch
import torch.nn as nn


class SpatioTemporalLSTMCell_plus(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCell_plus, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 7, width, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, width, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, width, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden * 2, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 2, width, width])
            )
            self.conv_c = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, width, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden * 2, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_c = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False)
            )

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        c_t_1_concat = self.conv_c(c_t)
        m_concat = self.conv_m(m_t)

        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_c_t_1, f_c_t_1, g_c_t_1 = torch.split(c_t_1_concat, self.num_hidden, dim=1)
        i_m_t_1, f_m_t_1, g_m_t_1, m_m_t_1 = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h + i_c_t_1)
        f_t = torch.sigmoid(f_x + f_h + f_c_t_1)
        g_t = torch.tanh(g_x + g_h + g_c_t_1)

        c_new = f_t * c_t + i_t * g_t
        c_concat = self.conv_c(c_new)

        i_c_prime, f_c_prime, g_c_prime = torch.split(c_concat, self.num_hidden, dim=1)

        i_t_prime = torch.sigmoid(i_x_prime + i_c_prime + i_m_t_1)
        f_t_prime = torch.sigmoid(f_x_prime + f_c_prime + f_m_t_1)
        g_t_prime = torch.tanh(g_x_prime + g_c_prime + g_m_t_1)

        m_new = f_t_prime * torch.tanh(m_m_t_1) + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_concat = self.conv_o(mem)
        o_mem, h_mem = torch.split(o_concat, self.num_hidden, dim=1)
        o_t = torch.tanh(o_x + o_mem)
        h_new = o_t * torch.tanh(h_mem)

        return h_new, c_new, m_new