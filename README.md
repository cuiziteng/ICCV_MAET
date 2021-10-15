# (ICCV 2021) Multitask AET with Orthogonal Tangent Regularity for Dark Object Detection [(paper)](https://openaccess.thecvf.com/content/ICCV2021/papers/Cui_Multitask_AET_With_Orthogonal_Tangent_Regularity_for_Dark_Object_Detection_ICCV_2021_paper.pdf) [(supp)](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Cui_Multitask_AET_With_ICCV_2021_supplemental.pdf)

**When Human Vision Meets Machine Vision (compare with enhancement methods):** <br/>
<img src="pics/example.jpg" height="250"> 

**Physics-based low-light degrading transformation (unprocess -- degradation -- ISP):**
<img src="pics/pipeline.jpg" height="250">

## Enviroment
```
python 3.7
pytorch 1.6.0
mmcv 1.1.5
matplotlib opencv-python Pillow tqdm
```
## Pre-trained Model
|  dataset   | model  | size | logs |
|  ----  | ----  | ----  | ----  |
| COCO  | ([google drive](https://drive.google.com/file/d/1thYimz_ciMFaZ03ICv61NfFZNnzRbcPN/view?usp=sharing)) ([baiduyun](https://pan.baidu.com/s/1A79a9377A7_zjf-vQYRHdw), passwd:1234) | 489.10 MB | - |
| EXDark  | ([google drive](https://drive.google.com/file/d/1thYimz_ciMFaZ03ICv61NfFZNnzRbcPN/view?usp=sharing)) ([baiduyun](https://pan.baidu.com/s/1Mrh_sOzXHhDo3Bk3inMiOg), passwd:1234) | 470.26 MB | [google drive](https://drive.google.com/file/d/1jU6lcjfQ5DuxThzGX2A_e-bPdBzJKaAT/view?usp=sharing) |

## Pre-process
**Step-1:** Download [COCO 2017 dataset](https://cocodataset.org/#download) and download [EXDark dataset] (include EXDark enhancement by MBLLEN, Zero-DCE, KIND) in VOC fashion from [google drive]() [baiduyun](). The EXDark dataset should be look like:
```
EXDark
│      
│
└───JPEGImages
│   │───IMGS (original low light)
│   │───IMGS_Kind (imgs enhancement by [Kind, mm 2019](https://arxiv.org/abs/1905.04161))
│   │───IMGS_ZeroDCE (imgs enhancement by [ZeroDCE, cvpr 2020](https://arxiv.org/abs/2001.06826))
│   │───IMGS_MEBBLN (imgs enhancement by [MEBBLN, bmvc 2018](http://bmvc2018.org/contents/papers/0700.pdf))
│───Annotations   
│───main
│───label
```

**Step-2:** Change the data place [line1](https://github.com/cuiziteng/MAET/blob/e7a23bce5cbfc089aafff205afa402f75823706e/configs/MAET_yolo/maet_yolo_exdark.py#L56) and [line2](https://github.com/cuiziteng/MAET/blob/e7a23bce5cbfc089aafff205afa402f75823706e/configs/MAET_yolo/maet_yolo_coco_ort.py#L63) to your own COCO and EXDark path.

## Testing
Testing MAET on (low-light) COCO dataset
```
python tools/test.py configs/MAET_yolo/maet_yolo_coco_ort.py [COCO model path] --eval bbox --show-dir [save dir]
```
Testing MAET on EXDark dataset
```
python tools/test.py configs/MAET_yolo/maet_yolo_exdark.py  [EXDark model path] --eval mAP --show-dir [save dir]
```

**Comparative Experiment** <br/>
Testing YOLOV3 on EXDark dataset enhancement by MEBBLN/ Kind/ Zero-DCE
```
python tools/test.py configs/MAET_yolo/yolo_mbllen.py (yolo_kind.py, yolo_zero_dce.py)  [MEBBLN/ Kind/ Zero-DCE model] --eval mAP --show-dir [save dir]
```

## Training
**Setp-1:** Pre-train MAET-COCO model (273 epochs on 4 GPUs): (if use other GPU number, please reset learining rate)
```
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=[port number] bash ./tools/dist_train_maet.sh configs/MAET_yolo/maet_yolo_coco_ort.py 4
```
**Setp-2:** Fine-tune on EXDark datastet (25epoch on 1 GPU): 
```
python tools/train.py configs/MAET_yolo/maet_yolo_exdark.py --gpu-ids [gpu id] --load-from [COCO model path]
```

**Comparative Experiment** <br/>
**Step-Addition:** Fine-tune EXDark dataset enhancement by MEBBLN/ Kind/ Zero-DCE (25epoch on 1 GPU, finetune on **well-trained normal COCO model**)
```
python tools/train.py configs/MAET_yolo/yolo_mbllen.py (yolo_kind.py, yolo_zero_dce.py) --gpu-ids [gpu id]
```

### Newly MAET-YOLO results on EXDark dataset (0.776 more than our paper's results):

| class     | gts  | dets | recall | ap    |
|  ----  | ----  | ----  | ----  | ----  |
| Bicycle   | 212  | 773  | 0.920  | 0.831 |
| Boat      | 289  | 942  | 0.900  | 0.785 |
| Bottle    | 282  | 1217 | 0.879  | 0.756 |
| Bus       | 135  | 331  | 0.970  | 0.929 |
| Car       | 597  | 1788 | 0.915  | 0.831 |
| Cat       | 183  | 579  | 0.885  | 0.734 |
| Chair     | 466  | 2132 | 0.854  | 0.713 |
| Cup       | 366  | 1086 | 0.880  | 0.790 |
| Dog       | 207  | 631  | 0.918  | 0.798 |
| Motorbike | 233  | 946  | 0.884  | 0.772 |
| People    | 1562 | 4353 | 0.906  | 0.811 |
| Table     | 333  | 1880 | 0.805  | 0.570 |
| mAP       |      |      |        | **0.777** |


## Citation
If our work help to your research, please cite our paper~ ^-^, thx.
```
@InProceedings{Cui_2021_ICCV,
    author    = {Cui, Ziteng and Qi, Guo-Jun and Gu, Lin and You, Shaodi and Zhang, Zenghui and Harada, Tatsuya},
    title     = {Multitask AET With Orthogonal Tangent Regularity for Dark Object Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {2553-2562}
}
```

**The code is largely borrow from mmdetection and unprocess, Thx to their wonderful works~** <br/>
MMdetection: [mmdetection](https://mmdetection.readthedocs.io/en/latest/) ([v2.7.0](https://github.com/open-mmlab/mmdetection/tree/v2.7.0)) <br/>
Unprocessing Images for Learned Raw Denoising: [unprocess](https://github.com/timothybrooks/unprocessing)
