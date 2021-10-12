import math
import numpy as np

import torch
import torch.nn as nn

from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      kaiming_init)
from mmcv.runner import load_checkpoint
from mmdet.utils import get_root_logger
from torch.nn.modules.batchnorm import _BatchNorm
from ..builder import BACKBONES


# build the default conv2d block
def conv2d(in_channels, out_channels, kernel_size, bias=True):
    # return nn.Conv2d(
    #     in_channels, out_channels, kernel_size,
    #     padding=(kernel_size//2), bias=bias)
    return build_conv_layer(None, in_channels, out_channels,
            kernel_size, padding=(kernel_size//2), bias=bias)

# mean shift process
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

# up sampling module: conv + pixel shuffle + bn + activation
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

#print(Upsampler(conv2d, 2, 16 * pow(2, 2), act=False))

            #    conv2d(16 * pow(2, 2), 16 * pow(2, 1), kernel_size=1))

# down sampling module: down conv + leakey relu + conv
class DownBlock(nn.Module):
    def __init__(self, scale, nFeat=16, in_channels=3, out_channels=3):
        super(DownBlock, self).__init__()
        # down sampling
        dual_block = [
            nn.Sequential(
                nn.Conv2d(in_channels, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        ]

        for _ in range(1, int(np.log2(scale))):
            dual_block.append(
                nn.Sequential(
                    nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                )
            )
        #conv
        dual_block.append(nn.Conv2d(nFeat, out_channels, kernel_size=3, stride=1, padding=1, bias=False))

        self.dual_module = nn.Sequential(*dual_block)

    def forward(self, x):
        x = self.dual_module(x)
        return x

###### The RCAB block proposed by YunFu's team in CVPR2018 ###### 
## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):  #channel should be 16's multiple
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

# print(RCAB(conv2d, 16 * pow(2, 2), 3, act=nn.ReLU(True)))
######################################################################################
###### The SR structure in our work, the code are inspired from DRN (cvpr 2020) ######
######################################################################################
@BACKBONES.register_module()
class DRN(nn.Module):
    def __init__(self, conv=conv2d, n_blocks = 16): #n_blocks: number of residual blocks, 16|30|40|80
        super(DRN, self).__init__()
        self.scale = [2]    # set to 2 for memory efficiency
        self.phase = len(self.scale)    #1  
        n_feats = 16
        kernel_size = 3

        act = nn.ReLU(True)
        #act = False

        # self.upsample = nn.Upsample(scale_factor=max(self.scale),
        #                             mode='bicubic', align_corners=False)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(255, rgb_mean, rgb_std)

        self.head = conv(3, n_feats, kernel_size)

        self.down = [
            DownBlock(2, n_feats * pow(2, p), n_feats * pow(2, p), n_feats * pow(2, p + 1)
            ) for p in range(self.phase)
        ]

        self.down = nn.ModuleList(self.down)

        up_body_blocks = [[
            RCAB(conv, n_feats * pow(2, p), kernel_size, act=act) for _ in range(n_blocks)]
             for p in range(self.phase, 1, -1)]

        up_body_blocks.insert(0, [
            RCAB(
                conv, n_feats * pow(2, self.phase), kernel_size, act=act
            ) for _ in range(n_blocks)
        ])

        # The fisrt upsample block
        up = [[Upsampler(conv, 2, n_feats * pow(2, self.phase), act=False),
               conv(n_feats * pow(2, self.phase), n_feats * pow(2, self.phase - 1), kernel_size=1)]]

        # The rest upsample blocks
        for p in range(self.phase - 1, 0, -1):
            up.append([Upsampler(conv, 2, 2 * n_feats * pow(2, p), act=False),
                conv(2 * n_feats * pow(2, p), n_feats * pow(2, p - 1), kernel_size=1)])

        self.up_blocks = nn.ModuleList()
        for idx in range(self.phase):
            self.up_blocks.append(
                nn.Sequential(*up_body_blocks[idx], *up[idx])
            )

        # tail conv that output sr imgs
        tail = [conv(n_feats * pow(2, self.phase), 3, kernel_size)]
        for p in range(self.phase, 0, -1):
            tail.append(
                conv(n_feats * pow(2, p), 3, kernel_size)
            )
        self.tail = nn.ModuleList(tail)

        self.add_mean = MeanShift(255, rgb_mean, rgb_std, 1)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        # preprocess
        x = self.sub_mean(x)
        x = self.head(x)

        # down phases,
        copies = []
        for idx in range(self.phase):
            copies.append(x)
            x = self.down[idx](x)

        # up phases
        sr = self.tail[0](x)
        sr = self.add_mean(sr)
        results = [sr]
        for idx in range(self.phase):
            # upsample to SR features
            x = self.up_blocks[idx](x)
            # concat down features and upsample features
            x = torch.cat((x, copies[self.phase - idx - 1]), 1)
            # output sr imgs
            sr = self.tail[idx + 1](x)
            sr = self.add_mean(sr)

            results.append(sr)

        return results
        # return x



# #print(DRN(conv2d).parameters())
# print([p.shape for p in DRN(conv2d).parameters()])
# tensor1 = torch.rand([1,3,224,224])
# # print(tensor1.shape)
# outs = DRN(conv2d).forward(tensor1)
# i = 0
# for out in outs:
#     i+=1
#     print(i)
#     print(out.shape)
# #print(.shape)

