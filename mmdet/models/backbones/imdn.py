import torch
import torch.nn as nn
from collections import OrderedDict

from ..builder import BACKBONES
from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      kaiming_init)
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.runner import load_checkpoint
from mmdet.utils import get_root_logger

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, dilation=dilation,
                     groups=groups)


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

# contrast-aware channel attention module
class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class IMDModule(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDModule, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = conv_layer(in_channels, in_channels, 3)
        self.c2 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(in_channels, in_channels, 1)
        self.cca = CCALayer(self.distilled_channels * 4)

    def forward(self, input):
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c4 = self.c4(remaining_c3)
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(self.cca(out)) + input
        return out_fused

class IMDModule_speed(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDModule_speed, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = conv_layer(in_channels, in_channels, 3)
        self.c2 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.distilled_channels * 4, in_channels, 1)

    def forward(self, input):
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c4 = self.c4(remaining_c3)

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(out) + input
        return out_fused

class IMDModule_Large(nn.Module):
    def __init__(self, in_channels, distillation_rate=1/4):
        super(IMDModule_Large, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)  # 6
        self.remaining_channels = int(in_channels - self.distilled_channels)  # 18
        self.c1 = conv_layer(in_channels, in_channels, 3, bias=False)  # 24 --> 24
        self.c2 = conv_layer(self.remaining_channels, in_channels, 3, bias=False)  # 18 --> 24
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3, bias=False)  # 18 --> 24
        self.c4 = conv_layer(self.remaining_channels, self.remaining_channels, 3, bias=False)  # 15 --> 15
        self.c5 = conv_layer(self.remaining_channels-self.distilled_channels, self.remaining_channels-self.distilled_channels, 3, bias=False)  # 10 --> 10
        self.c6 = conv_layer(self.distilled_channels, self.distilled_channels, 3, bias=False)  # 5 --> 5
        self.act = activation('relu')
        self.c7 = conv_layer(self.distilled_channels * 6, in_channels, 1, bias=False)

    def forward(self, input):
        out_c1 = self.act(self.c1(input))  # 24 --> 24
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1) # 6, 18
        out_c2 = self.act(self.c2(remaining_c1))  #  18 --> 24
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)  # 6, 18
        out_c3 = self.act(self.c3(remaining_c2))  # 18 --> 24
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)  # 6, 18
        out_c4 = self.act(self.c4(remaining_c3))  # 18 --> 18
        distilled_c4, remaining_c4 = torch.split(out_c4, (self.distilled_channels, self.remaining_channels-self.distilled_channels), dim=1)  # 6, 12
        out_c5 = self.act(self.c5(remaining_c4))  # 12 --> 12
        distilled_c5, remaining_c5 = torch.split(out_c5, (self.distilled_channels, self.remaining_channels-self.distilled_channels*2), dim=1)  # 6, 6
        out_c6 = self.act(self.c6(remaining_c5))  # 6 --> 6

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, distilled_c4, distilled_c5, out_c6], dim=1)
        out_fused = self.c7(out) + input
        return out_fused
    
def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)

def load_state_dict(path):
    
    state_dict = torch.load(path)
    new_state_dcit = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:]
        else:
            name = k
        new_state_dcit[name] = v
    return new_state_dcit

# For any upscale factors, we choose it as our SR backbone
@BACKBONES.register_module()
class IMDN_AS(nn.Module):
    def __init__(self, in_nc=3, nf=64, num_modules=6, out_nc=3, upscale=4):
        super(IMDN_AS, self).__init__()

        self.fea_conv = nn.Sequential(conv_layer(in_nc, nf, kernel_size=3, stride=2),
                                      nn.LeakyReLU(0.05),
                                      conv_layer(nf, nf, kernel_size=3, stride=2))

        # IMDBs
        self.IMDB1 = IMDModule(in_channels=nf)
        self.IMDB2 = IMDModule(in_channels=nf)
        self.IMDB3 = IMDModule(in_channels=nf)
        self.IMDB4 = IMDModule(in_channels=nf)
        self.IMDB5 = IMDModule(in_channels=nf)
        self.IMDB6 = IMDModule(in_channels=nf)

        # self.encoder = nn.Sequential(self.fea_conv(),self.IMDB1(),self.IMDB2(),self.IMDB3())
        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')
        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        upsample_block = pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            # model_dict = load_state_dict(pretrained)
            # print(model_dict)
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


    def forward(self, input, ext_fea=False):
        # print(input.shape)
        out_fea = self.fea_conv(input)
        out_B1 = self.IMDB1(out_fea)
        out_B2 = self.IMDB2(out_B1)
        out_B3 = self.IMDB3(out_B2)

        # # only extract feature for AET network
        if ext_fea:
            return out_B3

        # utilize the full sr network 
        else:
            out_B4 = self.IMDB4(out_B3)
            out_B5 = self.IMDB5(out_B4)
            out_B6 = self.IMDB6(out_B5)


            out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))
            out_lr = self.LR_conv(out_B) + out_fea
            output = self.upsampler(out_lr)
            return output

# # Set the IMDN as the encoder-decoder strcuture
# # @BACKBONES.register_module()
# class IMDN_AS_ED(nn.Module):
#     def __init__(self, in_nc=3, nf=64, num_modules=6, out_nc=3, upscale=4):
#         super(IMDN_AS, self).__init__()

#         self.fea_conv = nn.Sequential(conv_layer(in_nc, nf, kernel_size=3, stride=2),
#                                       nn.LeakyReLU(0.05),
#                                       conv_layer(nf, nf, kernel_size=3, stride=2))

#         # IMDBs
#         self.IMDB1 = IMDModule(in_channels=nf)
#         self.IMDB2 = IMDModule(in_channels=nf)
#         self.IMDB3 = IMDModule(in_channels=nf)
#         self.IMDB4 = IMDModule(in_channels=nf)
#         self.IMDB5 = IMDModule(in_channels=nf)
#         self.IMDB6 = IMDModule(in_channels=nf)
#         self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')
        
#         self.LR_conv = conv_layer(nf, nf, kernel_size=3)

#         self.encoder = nn.Sequential(self.IMDB1, self.IMDB2, self.IMDB3)

#         upsample_block = pixelshuffle_block
#         self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)
    
#     def init_weights(self, pretrained=None):
#         """Initialize the weights in backbone.

#         Args:
#             pretrained (str, optional): Path to pre-trained weights.
#                 Defaults to None.
#         """
#         if isinstance(pretrained, str):
#             # model_dict = load_state_dict(pretrained)
#             # print(model_dict)
#             logger = get_root_logger()
#             load_checkpoint(self, pretrained, strict=False, logger=logger)
#         elif pretrained is None:
#             for m in self.modules():
#                 if isinstance(m, nn.Conv2d):
#                     kaiming_init(m)
#                 elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
#                     constant_init(m, 1)
#         else:
#             raise TypeError('pretrained must be a str or None')


#     def forward(self, input):
#         # print(input.shape)
#         out_fea = self.fea_conv(input)
#         out_B1 = self.IMDB1(out_fea)
#         out_B2 = self.IMDB2(out_B1)
#         out_B3 = self.IMDB3(out_B2)
#         out_B4 = self.IMDB4(out_B3)
#         out_B5 = self.IMDB5(out_B4)
#         out_B6 = self.IMDB6(out_B5)


#         out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))
#         out_lr = self.LR_conv(out_B) + out_fea
#         output = self.upsampler(out_lr)
#         return output

class IMDN(nn.Module):
    def __init__(self, in_nc=3, nf=64, num_modules=6, out_nc=3, upscale=4):
        super(IMDN, self).__init__()

        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)

        # IMDBs
        self.IMDB1 = IMDModule(in_channels=nf)
        self.IMDB2 = IMDModule(in_channels=nf)
        self.IMDB3 = IMDModule(in_channels=nf)
        self.IMDB4 = IMDModule(in_channels=nf)
        self.IMDB5 = IMDModule(in_channels=nf)
        self.IMDB6 = IMDModule(in_channels=nf)
        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        upsample_block = pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)


    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.IMDB1(out_fea)
        out_B2 = self.IMDB2(out_B1)
        out_B3 = self.IMDB3(out_B2)
        out_B4 = self.IMDB4(out_B3)
        out_B5 = self.IMDB5(out_B4)
        out_B6 = self.IMDB6(out_B5)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea
        output = self.upsampler(out_lr)
        return output

# AI in RTC Image Super-Resolution Algorithm Performance Comparison Challenge (Winner solution)
class IMDN_RTC(nn.Module):
    def __init__(self, in_nc=3, nf=12, num_modules=5, out_nc=3, upscale=2):
        super(IMDN_RTC, self).__init__()

        fea_conv = [conv_layer(in_nc, nf, kernel_size=3)]
        rb_blocks = [IMDModule_speed(in_channels=nf) for _ in range(num_modules)]
        LR_conv = conv_layer(nf, nf, kernel_size=1)

        upsample_block = pixelshuffle_block
        upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)

        self.model = sequential(*fea_conv, ShortcutBlock(sequential(*rb_blocks, LR_conv)),
                                  *upsampler)

    def forward(self, input):
        output = self.model(input)
        return output


class IMDN_RTE(nn.Module):
    def __init__(self, upscale=2, in_nc=3, nf=20, out_nc=3):
        super(IMDN_RTE, self).__init__()
        self.upscale = upscale
        self.fea_conv = nn.Sequential(conv_layer(in_nc, nf, 3),
                                      nn.ReLU(inplace=True),
                                      conv_layer(nf, nf, 3, stride=2, bias=False))

        self.block1 = IMDModule_Large(nf)
        self.block2 = IMDModule_Large(nf)
        self.block3 = IMDModule_Large(nf)
        self.block4 = IMDModule_Large(nf)
        self.block5 = IMDModule_Large(nf)
        self.block6 = IMDModule_Large(nf)

        self.LR_conv = conv_layer(nf, nf, 1, bias=False)

        self.upsampler = pixelshuffle_block(nf, out_nc, upscale_factor=upscale**2)

    def forward(self, input):

        fea = self.fea_conv(input)
        out_b1 = self.block1(fea)
        out_b2 = self.block2(out_b1)
        out_b3 = self.block3(out_b2)
        out_b4 = self.block4(out_b3)
        out_b5 = self.block5(out_b4)
        out_b6 = self.block6(out_b5)

        out_lr = self.LR_conv(out_b6) + fea

        output = self.upsampler(out_lr)

        return output