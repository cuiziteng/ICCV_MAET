# (ICCV 2021) Multitask AET with Orthogonal Tangent Regularity for Dark Object Detection [(paper)](https://openaccess.thecvf.com/content/ICCV2021/papers/Cui_Multitask_AET_With_Orthogonal_Tangent_Regularity_for_Dark_Object_Detection_ICCV_2021_paper.pdf) [(supp)](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Cui_Multitask_AET_With_ICCV_2021_supplemental.pdf)

## Abstract

Dark environment becomes a challenge for computer vision algorithms owing to insufficient photons and undesirable noise. To enhance object detection in a dark environment, we propose a novel multitask auto encod- ing transformation (MAET) model which is able to explore the intrinsic pattern behind illumination translation. In a self-supervision manner, the MAET learns the intrinsic visual structure by encoding and decoding the realistic illumination-degrading transformation considering the physical noise model and image signal processing (ISP). Based on this representation, we achieve the object detection task by decoding the bounding box coordinates and classes. To avoid the over-entanglement of two tasks, our MAET disentangles the object and degrad- ing features by imposing an orthogonal tangent regularity. This forms a parametric manifold along which multi-task predictions can be geometrically formulated by maximizing the orthogonality between the tangents along the outputs of respective tasks. Our framework can be implemented based on the mainstream object detection ar- chitecture and directly trained end-to-end using normal target detection datasets, such as VOC and COCO. We have achieved the state-of-the-art performance using synthetic and real-world datasets.


**When Human Vision Meets Machine Vision (compare with enhancement methods):** <br/>
<img src="pics/example.jpg" height="350"> 

**Physics-based low-light degrading transformation (unprocess -- degradation -- ISP):**
<img src="pics/pipeline.jpg" height="350">

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
| **MAET-COCO (ours)** 80 class  | ([google drive](https://drive.google.com/file/d/1C7qntr0bW7piaNZPqPzpNZ0fIs8th-Qh/view?usp=sharing)) ([baiduyun](https://pan.baidu.com/s/1Mrh_sOzXHhDo3Bk3inMiOg), passwd:1234) | 489.10 MB | - |
| **MAET-EXDark (ours)** (77.7) 20 class | ([google drive](https://drive.google.com/file/d/1XCP4IgW579WlGljegDCjYpQA0V3vq7-E/view?usp=sharing)) ([baiduyun](https://pan.baidu.com/s/1rE0H1YPldj0ASBWmuksaIA), passwd:1234) | 470.26 MB | [google drive](https://drive.google.com/file/d/1jU6lcjfQ5DuxThzGX2A_e-bPdBzJKaAT/view?usp=sharing) |
| EXDark (76.8) 20 class | ([google drive]) ([baiduyun](https://pan.baidu.com/s/1WRqXA8-Tal7WFtIt-2v0jA), passwd:1234) | 470.26 MB | - |
| EXDark ([MBLLEN](http://bmvc2018.org/contents/papers/0700.pdf)) (76.3) 20 class| ([google drive](https://drive.google.com/drive/folders/1umRUBXEzHOSx1W1NDWpqrhuHXAQISviM?usp=sharing)) ([baiduyun](https://pan.baidu.com/s/161AxKd6aK3eGv2bl6UWMgg), passwd:1234) | 470.26 MB | - |
| EXDark ([Kind](https://arxiv.org/abs/1905.04161)) (76.3) 20 class| ([google drive](https://drive.google.com/drive/folders/1umRUBXEzHOSx1W1NDWpqrhuHXAQISviM?usp=sharing)) ([baiduyun](https://pan.baidu.com/s/1nav4RJcf8kF4CJU_CeAxjA), passwd:1234) | 470.26 MB | - |
| EXDark ([Zero-DCE](https://arxiv.org/abs/2001.06826)) (76.9) 20 class| ([google drive](https://drive.google.com/drive/folders/1umRUBXEzHOSx1W1NDWpqrhuHXAQISviM?usp=sharing)) ([baiduyun](https://pan.baidu.com/s/1rbl4Y26_sLAqcxj1bbDu2g), passwd:1234) | 470.26 MB | - |
| **MAET-UG2-DarkFace (ours)** (56.2) 1 class| ([google drive]()) ([baiduyun](https://pan.baidu.com/s/1vlvmVt_JFaWrUj2_YBNUTg), passwd:1234) | 469.81 MB | - |

## Pre-process
**Step-1:** 

For **MS COCO Dataset**: Download [COCO 2017 dataset](https://cocodataset.org/#download).

For **EXDark Dataset**: Download **EXDark** (include EXDark enhancement by MBLLEN, Zero-DCE, KIND) in VOC format from [google drive](https://drive.google.com/file/d/1X_zB_OSp_thhk9o26y1ZZ-F85UeS0OAC/view?usp=sharing) or [baiduyun](https://pan.baidu.com/s/1m4BMVqClhMks4S0xulkCcA), passwd:1234. The EXDark dataset should be look like:
```
EXDark
│      
│
└───JPEGImages
│   │───IMGS (original low light)
│   │───IMGS_Kind (imgs enhancement by [Kind, mm2019])
│   │───IMGS_ZeroDCE (imgs enhancement by [ZeroDCE, cvpr 2020])
│   │───IMGS_MEBBLN (imgs enhancement by [MEBBLN, bmvc 2018])
│───Annotations   
│───main
│───label
```

For **UG2-DarkFace Dataset**: Download **UG2** in VOC format from [google drive]() or [baiduyun](https://pan.baidu.com/s/1UfQq5ypVI9EYrBEin5225A), passwd:1234. The UG2-DarkFace dataset should be look like:
```
UG2
│      
└───main
│───xml  
│───label
│───imgs
```

**Step-2:** Cd in "your_project_path", and do set-up process (see [mmdetection](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation) if you want find details): 
```
git clone git@github.com:cuiziteng/ICCV_MAET.git
```
```
cd "your project path"
```
```
pip install -r requirements/build.txt
```
```
pip install -v -e .  # or "python setup.py develop"
```

**Step-3:** Change the data place [line1](https://github.com/cuiziteng/MAET/blob/e7a23bce5cbfc089aafff205afa402f75823706e/configs/MAET_yolo/maet_yolo_exdark.py#L56) and [line2](https://github.com/cuiziteng/MAET/blob/e7a23bce5cbfc089aafff205afa402f75823706e/configs/MAET_yolo/maet_yolo_coco_ort.py#L63) to your own COCO and EXDark path, and [line3](https://github.com/cuiziteng/ICCV_MAET/blob/4e0a530473b88f8958737e7beec566696ac56309/configs/MAET_yolo/maet_yolo_ug2.py#L56) to your own UG2-DarkFace path.

## Testing
Testing MAET-YOLOV3 on (low-light) COCO dataset
```
python tools/test.py configs/MAET_yolo/maet_yolo_coco_ort.py [COCO model path] --eval bbox --show-dir [save dir]
```
Testing MAET-YOLOV3 on EXDark dataset
```
python tools/test.py configs/MAET_yolo/maet_yolo_exdark.py  [EXDark model path] --eval mAP --show-dir [save dir]
```

Testing MAET-YOLOV3 on UG2-DarkFace dataset
```
python tools/test.py configs/MAET_yolo/maet_yolo_ug2.py [UG2-DarkFace model path] --eval mAP --show-dir [save dir]
```

**Comparative Experiment** <br/>
Testing YOLOV3 on EXDark dataset enhancement by MEBBLN/ Kind/ Zero-DCE
```
python tools/test.py configs/MAET_yolo/yolo_mbllen.py (yolo_kind.py, yolo_zero_dce.py)  [MEBBLN/ Kind/ Zero-DCE model] --eval mAP --show-dir [save dir]
```

## Training
**Setp-1:** Pre-train MAET-COCO model (273 epochs on 4 GPUs): (if use other GPU number, please reset learining rate), or direct download our pre-train COCO model ([google drive](https://drive.google.com/file/d/1C7qntr0bW7piaNZPqPzpNZ0fIs8th-Qh/view?usp=sharing)) ([baiduyun](https://pan.baidu.com/s/1Mrh_sOzXHhDo3Bk3inMiOg), passwd:1234).
```
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=[port number] bash ./tools/dist_train_maet.sh configs/MAET_yolo/maet_yolo_coco_ort.py 4
```
**Setp-2 (EXDark):** Fine-tune on EXDark datastet (25epoch on 1 GPU): 
```
python tools/train.py configs/MAET_yolo/maet_yolo_exdark.py --gpu-ids [gpu id] --load-from [COCO model path]
```
**Setp-2 (UG2-DarkFace):** Fine-tune on UG2-DarkFace datastet (20epoch on 1 GPU): 
```
python tools/train.py configs/MAET_yolo/maet_yolo_ug2.py --gpu-ids [gpu id] --load-from [COCO model path]
```

**Comparative Experiment** <br/>
Fine-tune EXDark dataset enhancement by MEBBLN/ Kind/ Zero-DCE (25epoch on 1 GPU) on well-trained normal COCO model (608x608) **for fairness**
```
python tools/train.py configs/MAET_yolo/yolo_mbllen.py (yolo_kind.py, yolo_zero_dce.py) --gpu-ids [gpu id]
```

### Baselines on EXDark dataset (renew), the baseline detector is YOLO-V3:

| class | Bicycle | Boat | Bottle | Bus | Car | Cat | Chair | Cup | Dog | Motorbike | People | Table | Total |
|  ---- | ----    | ---- | ----   | ----| ----| ----| ----  | ----| ----| ----      | ----   |  ---- |  ---- |
| Baseline |79.8 | 75.3 | 78.1 | 92.3 | 83.0 | 68.0 | 69.0 | 79.0 | 78.0 | 77.3 | 81.5 | 55.5 | 76.4 |
| KIND | 80.1 | 77.7 | 77.2 | 93.8 | 83.9 | 66.9 | 68.7 | 77.4 | 79.3 | 75.3 | 80.9 | 53.8 | 76.3 |
| MBLLEN | 82.0 | 77.3 | 76.5 | 91.3 | 84.0 | 67.6 | 69.1 | 77.6 | 80.4 | 75.6 | 81.9 | 58.6 | 76.8 |
| Zero-DCE | 84.1 | 77.6 | 78.3 | 93.1 | 83.7 | 70.3 | 69.8 | 77.6 | 77.4 | 76.3 | 81.0 | 53.6 | 76.9 |
| MAET (ours) | 83.1| 78.5| 75.6| 92.9| 83.1| 73.4| 71.3| 79.0| 79.8| 77.2| 81.1| 57.0| 77.7|
| [IAT-YOLOV3](https://github.com/cuiziteng/Illumination-Adaptive-Transformer) | 79.8 | 76.9 | 78.6 | 92.5 | 83.8 | 73.6 | 72.4 | 78.6 | 79.0 | 79.0 | 81.1 | 57.7 | 77.8 |



## Citation
If our work help to your research, please cite our paper, thx.
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
