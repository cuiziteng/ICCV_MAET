import logging

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Linear, constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES

# basic block consist of conv+bn+relu
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1):
        super(BasicBlock, self).__init__()
        padding = int((kernel_size-1)/2)
        self.layers = nn.Sequential()
        self.layers.add_module('Conv', nn.Conv2d(in_planes, out_planes,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        self.layers.add_module('BatchNorm', nn.BatchNorm2d(out_planes))
        self.layers.add_module('ReLU',      nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layers(x)

@BACKBONES.register_module()
class AETnet(nn.Module):
    '''
    The structure of AET (auto-encoding transformation) network,
    The paper: 'https://arxiv.org/pdf/1901.04596.pdf'.
    Input the original images and transformed images, 
    output the transformed informations.
    Here we adopt the thought of AET to boost our work.
    '''
    def __init__(self,
                in_channel = 3,
                feature1 = 64,
                feature2 = 128,
                conv_cfg = None,
                norm_cfg = dict(type='BN', requires_grad=True),
                act_cfg = dict(type='ReLU', inplace=True),
                reg_shape = 2,
                cls_type = 3):
        super(AETnet, self).__init__()

        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        
        self.encoder = nn.Sequential(ConvModule(in_channel, feature1, 3, 2, 1, **cfg),
                                     ConvModule(feature1, feature2, 3, 2, 1, **cfg),
                                     ConvModule(feature2, feature2, 3, 2, 1, **cfg))
        self.decoder_reg = nn.Sequential(nn.Linear(feature2*2, feature2), nn.BatchNorm1d(feature2), nn.ReLU(),
                                         nn.Linear(feature2, feature2//2), nn.BatchNorm1d(feature2//2), nn.ReLU(),
                                         nn.Linear(feature2//2, reg_shape))
        self.decoder_cls = nn.Sequential(nn.Linear(feature2*2, feature2), nn.BatchNorm1d(feature2), nn.ReLU(),
                                         nn.Linear(feature2, feature2//2), nn.BatchNorm1d(feature2//2), nn.ReLU(),
                                         nn.Linear(feature2//2, cls_type))
    
    
    # forward the two images: the original image 1 and the degradation image 2
    def forward(self, img1, img2):
        feature1 = self.encoder(img1)
        feature2 = self.encoder(img2)
        #print(feature1.shape)
        #print(feature2.shape)
        f_1, f_2 = self.global_pool(feature1), self.global_pool(feature2)
        #print(f_1.shape)
        #print(f_2.shape)
        f_aet = torch.cat((f_1, f_2), dim=1)
        #print(f_aet.shape)
        reg_pred = self.decoder_reg(f_aet)
        cls_pred = self.decoder_cls(f_aet)
        return reg_pred, cls_pred

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                if isinstance(m, nn.Linear):
                    #m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

        else:
            raise TypeError('pretrained must be a str or None')
    
    @staticmethod
    def global_pool(feat):
        num_channels = feat.size(1)
        return F.avg_pool2d(feat, (feat.size(2), feat.size(3))).view(-1, num_channels)

@BACKBONES.register_module()
class AETdecoder_Reg(nn.Module):
    '''
    The structure of AET (auto-encoding transformation) network's decoder,
    The paper: 'https://arxiv.org/pdf/1901.04596.pdf'.
    Input the original images' features and transformed images' features,
    output the transformed informations.
    Here we adopt the thought of AET to boost our work.
    '''

    def __init__(self, with_cls = False):
        # with_cls: do classification task or not
        super(AETdecoder_Reg, self).__init__()


        # self.decoder_reg = nn.Sequential(Linear(feature * 2, feature), nn.BatchNorm1d(feature), nn.ReLU(),
        #                                  Linear(feature, feature // 2), nn.BatchNorm1d(feature // 2), nn.ReLU(),
        #                                  Linear(feature // 2, reg_shape))
        num_inchannels = 64
        nChannels = 96
        nChannels2 = 80
        nChannels3 = 48
        num_classes = 2 # represent the scale factor and noise level
        self.with_cls = with_cls

        self.blocks1 = nn.Sequential(BasicBlock(num_inchannels, nChannels, 3), BasicBlock(nChannels, nChannels3, 1),
                                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.blocks2 = nn.Sequential(BasicBlock(nChannels3, nChannels, 3), BasicBlock(nChannels, nChannels, 1),
                                     nn.AvgPool2d(kernel_size=3, stride=2, padding=1))

        self.blocks3 = nn.Sequential(BasicBlock(nChannels, nChannels, 3), BasicBlock(nChannels, nChannels, 1))


        self.fc = nn.Linear(2*nChannels, num_classes)
        #self.fc_cls = nn.Linear(2*nChannels, 4)
        # self.weight_initialization()

    # forward the two images: the original image 1 and the degradation image 2
    def forward(self, feature1, feature2):
        # f_1, f_2 = self.global_pool(feature1), self.global_pool(feature2)
        # f_aet = torch.cat((f_1, f_2), dim=1)
        # reg_pred = self.decoder_reg(f_aet)
        # reg_pred = F.normalize(reg_pred,p=2,dim=1)
        feature1 = self.blocks3(self.blocks2(self.blocks1(feature1)))
        feature2 = self.blocks3(self.blocks2(self.blocks1(feature2)))

        f_1, f_2 = self.global_pool(feature1), self.global_pool(feature2)
        #print(f_1.shape)
        f_share = torch.cat((f_1, f_2), dim=1)
        #print(f_share.shape)
        reg_pred = self.fc(f_share)

        return torch.tanh(reg_pred)

    # def weight_initialization(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             if m.weight.requires_grad:
    #                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #                 m.weight.data.normal_(0, math.sqrt(2. / n))
    #         elif isinstance(m, nn.BatchNorm2d):
    #             if m.weight.requires_grad:
    #                 m.weight.data.fill_(1)
    #             if m.bias.requires_grad:
    #                 m.bias.data.zero_()
    #         elif isinstance(m, nn.Linear):
    #             if m.bias.requires_grad:
    #                 m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                if isinstance(m, nn.Linear):
                    #m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

        else:
            raise TypeError('pretrained must be a str or None')

    @staticmethod
    def global_pool(feat):
        num_channels = feat.size(1)
        return F.avg_pool2d(feat, (feat.size(2), feat.size(3))).view(-1, num_channels)

    @staticmethod
    def flatten(feat):
        batch_size = feat.size(0)
        return feat.view(feat.size(0), -1)


@BACKBONES.register_module()
class AETdecoder_dark(nn.Module):
    '''
    The structure of AET (auto-encoding transformation) network's decoder,
    The paper: 'https://arxiv.org/pdf/1901.04596.pdf'.
    Input the original images' features and transformed images' features,
    output the transformed informations.
    Here we adopt the thought of AET to boost our work.
    '''

    def __init__(self):
        # with_cls: do classification task or not
        super(AETdecoder_dark, self).__init__()

        # self.decoder_reg = nn.Sequential(Linear(feature * 2, feature), nn.BatchNorm1d(feature), nn.ReLU(),
        #                                  Linear(feature, feature // 2), nn.BatchNorm1d(feature // 2), nn.ReLU(),
        #                                  Linear(feature // 2, reg_shape))
        num_inchannels = 1024
        nChannels1 = 256
        nChannels2 = 128
        nChannels3 = 64
        outChannels = 48
        cls = 4

        self.blocks1 = nn.Sequential(BasicBlock(num_inchannels, nChannels1, 3), BasicBlock(nChannels1, nChannels2, 1),
                                     nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.blocks2 = nn.Sequential(BasicBlock(nChannels2, nChannels3, 3), BasicBlock(nChannels3, nChannels3, 1),
                                     nn.AvgPool2d(kernel_size=3, stride=2, padding=1))

        self.blocks3 = nn.Sequential(BasicBlock(nChannels3, outChannels, 3), BasicBlock(outChannels, outChannels, 1))

        # parameter prediction: darkness + others
        self.fc = nn.Linear(2 * outChannels, cls)
        self.weight_initialization()


    # forward the two images: the original image 1 and the degradation image 2
    def forward(self, feature1, feature2):

        feature1 = self.blocks3(self.blocks2(self.blocks1(feature1)))
        feature2 = self.blocks3(self.blocks2(self.blocks1(feature2)))
        # print('000', feature1.shape)
        f_1, f_2 = self.global_pool(feature1), self.global_pool(feature2)
        # print('111', f_1.shape)
        f_share = torch.cat((f_1, f_2), dim=1)
        # others prediction
        para_pred = self.fc(f_share)
        # print(para_pred.shape)

        return torch.tanh(para_pred)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight.requires_grad:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight.requires_grad:
                    m.weight.data.fill_(1)
                if m.bias.requires_grad:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias.requires_grad:
                    m.bias.data.zero_()

    # def init_weights(self, pretrained=None):
    #     if isinstance(pretrained, str):
    #         logger = logging.getLogger()
    #         load_checkpoint(self, pretrained, strict=False, logger=logger)
    #     elif pretrained is None:
    #         for m in self.modules():
    #             if isinstance(m, nn.Conv2d):
    #                 kaiming_init(m)
    #             if isinstance(m, nn.Linear):
    #                 #m.weight.data.normal_(0, 0.01)
    #                 m.bias.data.zero_()
    #             elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
    #                 constant_init(m, 1)
    #
    #     else:
    #         raise TypeError('pretrained must be a str or None')

    @staticmethod
    def global_pool(feat):
        num_channels = feat.size(1)
        return F.avg_pool2d(feat, (feat.size(2), feat.size(3))).view(-1, num_channels)

    @staticmethod
    def flatten(feat):
        batch_size = feat.size(0)
        return feat.view(feat.size(0), -1)