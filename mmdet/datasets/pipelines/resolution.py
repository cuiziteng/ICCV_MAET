import inspect

import os
import cv2
import matplotlib.pyplot as plt
import mmcv
import random
import numpy as np
import scipy
from scipy import ndimage

from ..builder import PIPELINES
import os.path as osp

def mkdir(paths):
    for path in paths:
        if not osp.exists(path):
            os.mkdir(path)


#########################################################################
#################### the code for resolution change  ####################
#########################################################################

############### following the degradation function in SR ################
# LR = ([HR * k]↓n + N) ↑n
# HR: original image, LR: generated low resolution image (same scale), n: down sampling scale, N: noise

@PIPELINES.register_module()
class Fixed_SR(object):

    def __init__(self,
                 scale = 2,
                 mode = 'same'
                 ):
        self.scale = scale
        #self.train_mode = train_mode
        self.ratio = 1/ self.scale
        self.mode = mode    # the mode could be chosen from 'same' (same scale) 'low' (low scale)

    def __call__(self, results):
        """Call function to perform dark light noise distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images after degradation process.
        """

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']
        #print(img.shape)
        ori_img = img.copy().astype(np.float32)
        ori_filename = results['ori_filename'].replace('JPEGImages/','')
        h, w, _ = img.shape
        dim = (int(w*self.ratio), int(h*self.ratio))

        noise_level_img = random.uniform(0, 10)/255.0   # random noise level
        kernel_width = random.uniform(0.2, 3.0) # random kernel width

        # blur kernel, gaussian blur kernel and motion blur kernel
        kernel = self.fspecial_gaussian(15, kernel_width)
        kernel /= np.sum(kernel)
        #print('kernel shape is', kernel.shape)
        img_blur = ndimage.filters.convolve(img, kernel[..., np.newaxis], mode='wrap')

        # bicubic downsampling
        img_down = cv2.resize(img_blur, dim, interpolation = cv2.INTER_CUBIC)
        #print('img dowm shape:', img_down.shape)
        # add noises
        noise = np.random.normal(0, noise_level_img, img_down.shape)
        #print('the noise is:', noise)
        #print('the noise shape is', noise.shape)
        #print(noise[:,:,0] - noise[:,:,1])
        #print(noise[:,:,0].shape)
        img_down += noise
        
        # bicubic upsampling
        img_low =  cv2.resize(img_down, (w, h), interpolation = cv2.INTER_CUBIC)
        #print('img low shape:', img_low.shape)

        # img_save_path = osp.join(save_path, ori_filename)
        # img_save_path1 = osp.join(save_path1, ori_filename)
        # img_save_path2 = osp.join(save_path2, ori_filename)
        # img = np.clip(img , 0.0, 1.0)
        # img_down = np.clip(img_down , 0.0, 1.0)
        # img_low = np.clip(img_low , 0.0, 1.0)
        # plt.imsave(img_save_path, img)
        # plt.imsave(img_save_path1, img_down)
        # plt.imsave(img_save_path2, img_low)
        #img_save_path3 = osp.join(save_path3, ori_filename)

        assert img.dtype == np.float32, \
            'This process needs the input image of dtype np.float32,'\
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
        
        # img = np.clip(img , 0.0, 1.0)
        
        # img = img.copy().astype(np.float32)

        if self.mode == 'same':
            #print('1111111111111')

            assert img_low.dtype == np.float32, \
                'This process needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            results['img'] = img_low
            results['img_shape'] = img_low.shape

            # for key in results.get('bbox_fields', []):
            #     bboxes = results[key]
            #     print(bboxes.shape)

        elif self.mode == 'low':
            assert img_down.dtype == np.float32, \
                'This process needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            img_down = img_down.copy().astype(np.float32)
            results['img'] = img_down
            results['img_shape'] = img_down.shape
            
            # change the bbox label to same scale
            for key in results.get('bbox_fields', []):
                # e.g. gt_bboxes and gt_bboxes_ignore
                bboxes = results[key]
                img_shape = results['img_shape']
                print('bbox before', bboxes)

                scale_factor = np.array([self.ratio, self.ratio, self.ratio, self.ratio],
                                    dtype=np.float32)
                bboxes = results[key] * scale_factor
                print('bbox after', bboxes)
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
                results[key] = bboxes

        

        # #lightness k, red gain, blue gain, gamma, noise var, quantization step
        # transforms = np.array([[k, red_gain, blue_gain, gamma, var, quan]]).astype(np.float32)
        # rgb2cam = np.reshape(rgb2cam, (-1,9)).astype(np.float32)    #3x3 matrix
        # #print(rgb2cam)
        # #transforms martrix and rgb2cam matrix 
        # results['transform'] = transforms
        # results['rgb2cam'] = rgb2cam
        return results
    
    # fixed gaussian kernel
    def fspecial_gaussian(self, hsize, sigma):
        
        hsize = [hsize, hsize]
        siz = [(hsize[0]-1.0)/2.0, (hsize[1]-1.0)/2.0]
        std = sigma
        [x, y] = np.meshgrid(np.arange(-siz[1], siz[1]+1), np.arange(-siz[0], siz[0]+1))
        arg = -(x*x + y*y)/(2*std*std)
        h = np.exp(arg)
        h[h < scipy.finfo(float).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h = h/sumh
        return h
        

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
'''
@PIPELINES.register_module()
class Random_SR(object):
    # generate a random resolution images in 1x to 4x (with degradation model)
    def __init__(self,
                 scale_range = (0.25, 1), 
                 ):
        self.scale_range = scale_range
        #self.train_mode = train_mode

    def __call__(self, results):
        """Call function to perform dark light noise distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images after degradation process.
        """

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']

        ori_img = img.copy().astype(np.float32)
        ori_filename = results['ori_filename'].replace('JPEGImages/','')
        h, w, _ = img.shape

        # random choose the down sampling scale
        ratio = random.uniform(self.scale_range[0], self.scale_range[1])
        dim = (int(w*ratio), int(h*ratio))

        noise_level_img = random.uniform(0, 10)/255.0   # random noise level
        kernel_width = random.uniform(0.2, 3.0) # random kernel width

        # blur kernel, gaussian blur kernel and motion blur kernel
        kernel = self.fspecial_gaussian(15, kernel_width)
        kernel /= np.sum(kernel)
        img_blur = ndimage.filters.convolve(img, kernel[..., np.newaxis], mode='wrap')

        # bicubic downsampling
        img_down = cv2.resize(img_blur, dim, interpolation = cv2.INTER_CUBIC)
        
        # add noises
        img_down += np.random.normal(0, noise_level_img, img_down.shape)

        # bicubic upsampling
        img_low =  cv2.resize(img_down, (w, h), interpolation = cv2.INTER_CUBIC)

        # img_save_path = osp.join(save_path, ori_filename)
        # img_save_path1 = osp.join(save_path1, ori_filename)
        # img_save_path2 = osp.join(save_path2, ori_filename)
        # img = np.clip(img , 0.0, 1.0)
        # img_down = np.clip(img_down , 0.0, 1.0)
        # img_low = np.clip(img_low , 0.0, 1.0)
        # plt.imsave(img_save_path, img)
        # plt.imsave(img_save_path1, img_down)
        # plt.imsave(img_save_path2, img_low)
        #img_save_path3 = osp.join(save_path3, ori_filename)

        assert img.dtype == np.float32, \
            'This process needs the input image of dtype np.float32,'\
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
        
        # img = np.clip(img , 0.0, 1.0)
        
        # img = img.copy().astype(np.float32)

        
        assert img_down.dtype == np.float32, \
            'This process needs the input image of dtype np.float32,'\
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
        results['img'] = img_down
        results['img_shape'] = img_down.shape
        
        # change the bbox label to same scale
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bboxes = results[key]
            img_shape = results['img_shape']
            print('bbox before', bboxes)

            scale_factor = np.array([self.ratio, self.ratio, self.ratio, self.ratio],
                                dtype=np.float32)
            bboxes = results[key] * scale_factor
            print('bbox after', bboxes)
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes

        

        # #lightness k, red gain, blue gain, gamma, noise var, quantization step
        # transforms = np.array([[k, red_gain, blue_gain, gamma, var, quan]]).astype(np.float32)
        # rgb2cam = np.reshape(rgb2cam, (-1,9)).astype(np.float32)    #3x3 matrix
        # #print(rgb2cam)
        # #transforms martrix and rgb2cam matrix 
        # results['transform'] = transforms
        # results['rgb2cam'] = rgb2cam
        return results
    
    # fixed gaussian kernel
    def fspecial_gaussian(self, hsize, sigma):
        
        hsize = [hsize, hsize]
        siz = [(hsize[0]-1.0)/2.0, (hsize[1]-1.0)/2.0]
        std = sigma
        [x, y] = np.meshgrid(np.arange(-siz[1], siz[1]+1), np.arange(-siz[0], siz[0]+1))
        arg = -(x*x + y*y)/(2*std*std)
        h = np.exp(arg)
        h[h < scipy.finfo(float).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h = h/sumh
        return h
        

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
'''

