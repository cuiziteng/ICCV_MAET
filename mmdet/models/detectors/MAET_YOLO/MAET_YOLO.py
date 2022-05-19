# coding=utf-8
# Copyright 2020 Ziteng Cui, email: cuiziteng@sjtu.edu.cn.

import torch
import torch.nn as nn
import os
import os.path as osp
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import random
import scipy.stats as stats

from mmdet.core import bbox2result
from ...builder import DETECTORS, build_backbone, build_head, build_neck, build_shared_head, build_loss
from ..base import BaseDetector

# hook gradient
# grads = {}
# def save_grad(name):
#     def hook_fn(grad):
#         #print(grad)
#         grads[name] = grad
#         return grad
#     return hook_fn

def random_noise_levels():
    """Generates random shot and read noise from a log-log linear distribution."""
    log_min_shot_noise = np.log(0.0001)
    log_max_shot_noise = np.log(0.012)
    log_shot_noise = np.random.uniform(log_min_shot_noise, log_max_shot_noise)
    shot_noise = np.exp(log_shot_noise)

    line = lambda x: 2.18 * x + 1.20
    log_read_noise = line(log_shot_noise) + np.random.normal(scale=0.26)
    # print('shot noise and read noise:', log_shot_noise, log_read_noise)
    read_noise = np.exp(log_read_noise)
    return shot_noise, read_noise

# hook gradient
grads = {}
def save_grad(name):
    def hook_fn(grad):
        #print(grad)
        grads[name] = grad
        return grad
    return hook_fn

@DETECTORS.register_module()
class MAET_YOLO(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 aet=None,
                 ort_cfg=None,
                 degration_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(MAET_YOLO, self).__init__()

        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        # self.aet = build_backbone(aet)
        self.aet = build_shared_head(aet)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.degration_cfg = degration_cfg
        self.use_ori = ort_cfg['use_ort']
        self.init_weights(pretrained=pretrained)
        self.loss_ort = nn.CosineSimilarity(dim=1)


    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(MAET_YOLO, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def apply_ccm(self, image, ccm):
        '''
        The function of apply CCM matrix
        '''
        shape = image.shape
        image = image.view(-1, 3)
        image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
        return image.view(shape)

    def weight_L1_loss(self, pred, gt, weight=5):
        '''
        pred and gt are two tensors:
        pred: (B, 1+K) 1: darkness, K: other parameters
        gt: (B, 1+K) 1: darkness, K: other parameters
        '''
        loss = weight*torch.mean((pred[:, 0:1]-gt[:, 0:1])**2) + torch.mean((pred[:, 1:]-gt[:, 1:])**2)
        return loss

    def Low_Illumination_Degrading(self, img, img_meta, safe_invert=False):

        '''
        (1)unprocess part(RGB2RAW) (2)low light corruption part (3)ISP part(RAW2RGB)
        Some code copy from 'https://github.com/timothybrooks/unprocessing', thx to their work ~
        input:
        img (Tensor): Input normal light images of shape (C, H, W).
        img_meta(dict): A image info dict contain some information like name ,shape ...
        return:
        img_deg (Tensor): Output degration low light images of shape (C, H, W).
        degration_info(Tensor): Output degration paramter in the whole process.
        '''

        '''
        parameter setting
        '''
        device = img.device
        config = self.degration_cfg
        # camera color matrix
        xyz2cams = [[[1.0234, -0.2969, -0.2266],
                     [-0.5625, 1.6328, -0.0469],
                     [-0.0703, 0.2188, 0.6406]],
                    [[0.4913, -0.0541, -0.0202],
                     [-0.613, 1.3513, 0.2906],
                     [-0.1564, 0.2151, 0.7183]],
                    [[0.838, -0.263, -0.0639],
                     [-0.2887, 1.0725, 0.2496],
                     [-0.0627, 0.1427, 0.5438]],
                    [[0.6596, -0.2079, -0.0562],
                     [-0.4782, 1.3016, 0.1933],
                     [-0.097, 0.1581, 0.5181]]]
        rgb2xyz = [[0.4124564, 0.3575761, 0.1804375],
                   [0.2126729, 0.7151522, 0.0721750],
                   [0.0193339, 0.1191920, 0.9503041]]

        # noise parameters and quantization step

        '''
        (1)unprocess part(RGB2RAW): 1.inverse tone, 2.inverse gamma, 3.sRGB2cRGB, 4.inverse WB digital gains
        '''
        img1 = img.permute(1, 2, 0)  # (C, H, W) -- (H, W, C)
        # print(img1.shape)
        # img_meta = img_metas[i]
        # inverse tone mapping
        img1 = 0.5 - torch.sin(torch.asin(1.0 - 2.0 * img1) / 3.0)
        # inverse gamma
        epsilon = torch.FloatTensor([1e-8]).to(torch.device(device))
        gamma = random.uniform(config['gamma_range'][0], config['gamma_range'][1])
        img2 = torch.max(img1, epsilon) ** gamma
        # sRGB2cRGB
        xyz2cam = random.choice(xyz2cams)
        rgb2cam = np.matmul(xyz2cam, rgb2xyz)
        rgb2cam = torch.from_numpy(rgb2cam / np.sum(rgb2cam, axis=-1)).to(torch.float).to(torch.device(device))
        # print(rgb2cam)
        img3 = self.apply_ccm(img2, rgb2cam)
        # img3 = torch.clamp(img3, min=0.0, max=1.0)

        # inverse WB
        rgb_gain = random.normalvariate(config['rgb_range'][0], config['rgb_range'][1])
        red_gain = random.uniform(config['red_range'][0], config['red_range'][1])
        blue_gain = random.uniform(config['blue_range'][0], config['blue_range'][1])

        gains1 = np.stack([1.0 / red_gain, 1.0, 1.0 / blue_gain]) * rgb_gain
        # gains1 = np.stack([1.0 / red_gain, 1.0, 1.0 / blue_gain])
        gains1 = gains1[np.newaxis, np.newaxis, :]
        gains1 = torch.FloatTensor(gains1).to(torch.device(device))

        # color disorder !!!
        if safe_invert:
            img3_gray = torch.mean(img3, dim=-1, keepdim=True)
            inflection = 0.9
            zero = torch.zeros_like(img3_gray).to(torch.device(device))
            mask = (torch.max(img3_gray - inflection, zero) / (1.0 - inflection)) ** 2.0
            safe_gains = torch.max(mask + (1.0 - mask) * gains1, gains1)

            #img4 = img3 * gains1
            img4 = torch.clamp(img3*safe_gains, min=0.0, max=1.0)

        else:
            img4 = img3 * gains1

        '''
        (2)low light corruption part: 5.darkness, 6.shot and read noise 
        '''
        # darkness(low photon numbers)
        lower, upper = config['darkness_range'][0], config['darkness_range'][1]
        mu, sigma = 0.1, 0.08
        darkness = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        darkness = darkness.rvs()
        # print(darkness)
        img5 = img4 * darkness
        # add shot and read noise
        shot_noise, read_noise = random_noise_levels()
        var = img5 * shot_noise + read_noise  # here the read noise is independent
        var = torch.max(var, epsilon)
        # print('the var is:', var)
        noise = torch.normal(mean=0, std=torch.sqrt(var))
        img6 = img5 + noise

        '''
        (3)ISP part(RAW2RGB): 7.quantisation  8.white balance 9.cRGB2sRGB 10.gamma correction
        '''
        # quantisation noise: uniform distribution
        bits = random.choice(config['quantisation'])
        quan_noise = torch.FloatTensor(img6.size()).uniform_(-1 / (255 * bits), 1 / (255 * bits)).to(
            torch.device(device))
        # print(quan_noise)
        # img7 = torch.clamp(img6 + quan_noise, min=0)
        img7 = img6 + quan_noise
        # white balance
        gains2 = np.stack([red_gain, 1.0, blue_gain])
        gains2 = gains2[np.newaxis, np.newaxis, :]
        gains2 = torch.FloatTensor(gains2).to(torch.device(device))
        img8 = img7 * gains2
        # cRGB2sRGB
        cam2rgb = torch.inverse(rgb2cam)
        img9 = self.apply_ccm(img8, cam2rgb)
        # gamma correction
        img10 = torch.max(img9, epsilon) ** (1 / gamma)



        img_low = img10.permute(2, 0, 1)  # (H, W, C) -- (C, H, W)
        # degration infomations: darkness, gamma value, WB red, WB blue
        # dark_gt = torch.FloatTensor([darkness]).to(torch.device(device))
        para_gt = torch.FloatTensor([darkness, 1.0 / gamma, 1.0 / red_gain, 1.0 / blue_gain]).to(torch.device(device))
        # others_gt = torch.FloatTensor([1.0 / gamma, 1.0, 1.0]).to(torch.device(device))
        # print('the degration information:', degration_info)
        return img_low, para_gt

    def extract_feat_aet(self, img, img_dark):
        """
        Extract features of normal light images and low light images.
        img --> backbone --> light_feature
        low_img --> backbone --> low_feature1 --> neck --> low_feature2
        concat(light_feature, low_feature1) --> aet_head
        """
        x_light = self.backbone(img)
        x_dark = self.backbone(img_dark)
        feat = x_light[2]  # last feature in backbone of low light images
        feat1 = x_dark[2]  # last feature in backboe of high light images

        para_pred = self.aet(feat, feat1)
        # para_pred.register_hook(save_grad('aet_grad'))
        x_light[2].register_hook(save_grad('light_grad'))
        x_dark[2].register_hook(save_grad('dark_grad'))

        if self.with_neck:
            x_dark = self.neck(x_dark)
        # print(x_trans.shape)
        return x_dark, para_pred

    def extract_feat(self, img_dark):
        '''
        Only low light images were used for validation and test.
        '''
        x_dark = self.backbone(img_dark)
        if self.with_neck:
            x_dark = self.neck(x_dark)
        return x_dark

    def forward_dummy(self, img, img_dark):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x, _ = self.extract_feat(img, img_dark)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # print(img.shape)
        # generate low light degration images part and get degration informations
        # generate low light images
        batch_size = img.shape[0]
        device = img.device
        img_dark = torch.empty(size=(batch_size, img.shape[1], img.shape[2], img.shape[3])).to(torch.device(device))
        para_gt = torch.empty(size=(batch_size, 4)).to(torch.device(device))
        # others_gt = torch.empty(size=(batch_size, 3)).to(torch.device(device))

        # Generation of degraded data and AET groundtruth
        for i in range(batch_size):
            img_dark[i], para_gt[i] = self.Low_Illumination_Degrading(img[i], img_metas[i])
        # img_dark = torch.stack([self.Low_Illumination_Degrading(img[i], img_metas[i])[0] for i in range(img.shape[0])],
        #                        dim=0)
        # degration_info = torch.stack(
        #     [self.Low_Illumination_Degrading(img[i], img_metas[i])[1] for i in range(img.shape[0])], dim=0)
        # print(degration_info)
        # print(degration_info.shape)
        # print(img_dark.shape)
        x_dark, para_pred = self.extract_feat_aet(img, img_dark)
        losses = self.bbox_head.forward_train(x_dark, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        aet_loss = 10*self.weight_L1_loss(para_pred, para_gt)

        losses['loss_aet'] = [aet_loss]
        #losses.update(aet_loss)
        if not grads:
            #print('0000000000')
           ort_loss = torch.cuda.FloatTensor(1).fill_(0.0).to(torch.device(device))
        if grads:
            #print('1111111111')
            ort_loss = 5*torch.mean(torch.abs(self.loss_ort(grads['light_grad'].view(batch_size,-1), grads['dark_grad'].view(batch_size,-1))))+\
                            0.5*torch.mean(1 - torch.abs(self.loss_ort(grads['light_grad'].view(batch_size,-1), grads['light_grad'].view(batch_size,-1)))) +\
                            0.5*torch.mean(1 - torch.abs(self.loss_ort(grads['dark_grad'].view(batch_size,-1), grads['dark_grad'].view(batch_size,-1))))
           # ort_loss = 0.1 * torch.mean(torch.abs(self.loss_ort(grads['light_grad'].view(batch_size,-1),
           #                                                     grads['dark_grad'].view(batch_size,-1))))
        losses['loss_ort'] = ort_loss

        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        # print('0000',img.shape)
        img_dark = torch.stack([self.Low_Illumination_Degrading(img[i], img_metas[i])[0] for i in range(img.shape[0])], dim=0)
        # x = self.extract_feat_test(img_dark)
        x = self.extract_feat(img_dark)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        # print(imgs)
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'
        print('11111',imgs)
        # img_dark = torch.stack([self.Low_Illumination_Degrading(img[i], img_metas[i])[0]  for i in range(img.shape[0])], dim=0)
        # feate, _ = self.extract_feat(img, img_dark)
        feats = self.extract_feats(imgs)
        return [self.bbox_head.aug_test(feats, img_metas, rescale=rescale)]
