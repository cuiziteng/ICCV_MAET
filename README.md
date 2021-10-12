# (ICCV 2021) Multitask AET with Orthogonal Tangent Regularity for Dark Object Detection [(paper)](https://openaccess.thecvf.com/content/ICCV2021/papers/Cui_Multitask_AET_With_Orthogonal_Tangent_Regularity_for_Dark_Object_Detection_ICCV_2021_paper.pdf) [(supp)](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Cui_Multitask_AET_With_ICCV_2021_supplemental.pdf)

**When Human Vision Meets Machine Vision:**
<img src="pics/example.jpg" height="250"> <br/>

**Physics-based low-light degrading transformation (unprocess -- degradation -- ISP):**
<img src="pics/pipeline.jpg" height="250">

## Pre-trained Model
|  dataset   | model  | size | logs |
|  ----  | ----  | ----  | ----  |
| COCO  | ([google drive](https://drive.google.com/file/d/1thYimz_ciMFaZ03ICv61NfFZNnzRbcPN/view?usp=sharing)) ([baiduyun](https://pan.baidu.com/s/1A79a9377A7_zjf-vQYRHdw), passwd:1234) | | |
| EXDark  | ([google drive](https://drive.google.com/file/d/1thYimz_ciMFaZ03ICv61NfFZNnzRbcPN/view?usp=sharing)) ([baiduyun](https://pan.baidu.com/s/1Mrh_sOzXHhDo3Bk3inMiOg), passwd:1234) | | |

## Citation
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
