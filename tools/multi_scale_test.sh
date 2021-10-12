# python tools/test.py configs/retinanet/retinanet_r50_fpn_1x_coco.py retinanet_resnet50_fpn.pth --eval bbox
python tools/test.py configs/retinanet/scales/scale_down_2.py retinanet_resnet50_fpn.pth --eval bbox
python tools/test.py configs/retinanet/scales/scale_down_3.py retinanet_resnet50_fpn.pth --eval bbox
python tools/test.py configs/retinanet/scales/scale_down_4.py retinanet_resnet50_fpn.pth --eval bbox
python tools/test.py configs/retinanet/scales/scale_up1_5.py retinanet_resnet50_fpn.pth --eval bbox
python tools/test.py configs/retinanet/scales/scale_up2.py retinanet_resnet50_fpn.pth --eval bbox