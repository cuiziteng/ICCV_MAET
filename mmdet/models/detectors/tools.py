import torch
import torch.nn as nn
import numpy as np
import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import os.path as osp

matplotlib.use('agg')

VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor']


def visual_img(img, img_meta, show_path):
    '''
    The visualization the middle image.
    '''
    if not osp.exists(show_path):
        os.mkdir(show_path)

    img = img.cpu().numpy()

    # print(img_meta['ori_filename'])
    name = img_meta['ori_filename'].replace('JPEGImages/', '')
    # print(name)
    # img_single = np.transpose(img[i, :, :, :], (1, 2, 0))
    # print(img_single)
    img_single = np.clip(img, 0, 1) * 255.0
    img_single = cv2.UMat(img_single).get()

    img_single = img_single / 255.0

    plt.imsave(osp.join(show_path, name), img_single)

def visual_imgs(img, img_metas, show_path):
    '''
    The visualization the middle images.
    '''
    if not osp.exists(show_path):
        os.makedirs(show_path)

    img = img.cpu().numpy()
    
    for i in range(img.shape[0]):
        print(i)
        #print(img_meta['ori_filename'])
        name = img_metas[i]['ori_filename'].replace('JPEGImages/','')
        #print(name)
        img_single = np.transpose(img[i,:,:,:], (1,2,0))
        #print(img_single)
        img_single = np.clip(img_single, 0, 1)*255.0
        img_single = cv2.UMat(img_single).get()
        
        img_single = img_single/255.0

        plt.imsave(osp.join(show_path, name), img_single)

def check_locations(img, img_metas, gt_bboxes, gt_labels, show_path):
    '''
    The code to check if images and bounding box in the right location. By cuiziteng@sjtu.edu.cn
    '''
    if not osp.exists(show_path):
        os.makedirs(show_path)

    img = img.cpu().numpy()
    
    for i in range(img.shape[0]):
        #print(img_metas[i]['ori_filename'])
        name = img_metas[i]['ori_filename'].replace('JPEGImages/','')
        #print(name)
        img_single = np.transpose(img[i,:,:,:], (1,2,0))
        #print(img_single)
        img_single = np.clip(img_single, 0, 1)*255.0
        #print(np.shape(img_single))
        #print('1111111', gt_bboxes[i])
        #print('2222222', gt_labels[i])
        gt_bboxes1 = gt_bboxes[i].cpu().numpy()
        gt_labels1 = gt_labels[i].cpu().numpy()
        img_single = cv2.UMat(img_single).get()
        for k, gt_bbox in enumerate(gt_bboxes1):
            print('111', k, gt_bbox)
            # print('222', VOC_CLASSES[int(gt_labels1[k])])
            # label_text = VOC_CLASSES[int(gt_labels1[k])]
            xmin,ymin,xmax,ymax = int(gt_bbox[0]), int(gt_bbox[1]), int(gt_bbox[2]), int(gt_bbox[3])
            # locate box
            img_single = cv2.rectangle(np.array(img_single),(xmin, ymin), (xmax, ymax), (255,0,0), 2)
            # locate class
            # img_single = cv2.putText(img_single, label_text, (xmin, xmax - 2),
            #             cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255))
        
        img_single = img_single/255.0

        plt.imsave(osp.join(show_path, name), img_single)

# generate different type of kernels


        



