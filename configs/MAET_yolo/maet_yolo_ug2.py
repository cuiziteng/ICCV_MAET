_base_ = '../_base_/default_runtime.py'
load_from = '/home/czt/mmdetection_v1/work_dirs/maet_yolo_coco_ort/latest.pth'
# model settings
model = dict(
    type='YOLOV3',
    pretrained='open-mmlab://darknet53',
    backbone=dict(type='Darknet', depth=53, out_indices=(3, 4, 5)),
    neck=dict(
        type='YOLOV3Neck',
        num_scales=3,
        in_channels=[1024, 512, 256],
        out_channels=[512, 256, 128]),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=1,
        in_channels=[512, 256, 128],
        out_channels=[1024, 512, 256],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(58, 45), (78, 99), (180, 163)],
                        [(15, 31), (32, 25), (29, 59)],
                        [(5, 8), (8, 15), (23, 13)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16, 8],
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='sum'),
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum')))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='GridAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0))
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    conf_thr=0.005,
    nms=dict(type='nms', iou_thr=0.55),
    max_per_img=100)

# dataset settings
dataset_type = 'UG2FaceDataset'
data_root = '/home/czt/DataSets/Dark_face_2019/'
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=[(320, 320), (608, 608)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg), #this step change the images from bgr2rgb
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    #return img gt_bboxer gt_labels
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(664, 664),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=12,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file = data_root + 'main/train.txt',
            img_prefix=data_root,
            pipeline=train_pipeline)),
    #train=dict(
    #    type=dataset_type,
    #    ann_file=data_root + 'main/train.txt',
    #    img_prefix=data_root,
    #    pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file = data_root + 'main/val.txt',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file = data_root + 'main/val.txt',
        img_prefix=data_root,
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    #warmup='linear',
    #warmup_iters=2000,  # same as burn-in in darknet
    #warmup_ratio=0.1,
    step=[14, 18])
# runtime settings
total_epochs = 20
evaluation = dict(interval=1, metric=['mAP'])
