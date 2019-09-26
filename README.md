## Feature Selective Anchor-Free Module for Single-Shot Object Detection. CVPR, 2019. (in PyTorch)


## Description

This repository reproduces "Zhu et al. Feature Selective Anchor-Free Module for Single-Shot Object Detection. CVPR, 2019." (FSAF) [PDF](https://arxiv.org/pdf/1903.00621.pdf) in PyTorch. The implementation is based on MMDetection framework. All the codes for the FSAF model follow the original paper.


## Get Started

To use this repo, please follow [README.md](./README_MMDetection.md) of MMDetection.


## Train/Eval

**Train**
- To train baseline (i.e., RetinaNet)
```Shell
./tools/dist_train_retinanet_r50_400_050x.sh
```
- To train FSAF (w/o anchor-based (AB))
```Shell
./tools/dist_train_fsaf_r50_400_050x.sh
```
**Eval**

For evaluation, pretrained model-weights should be located at "./models/here".

- To evaluate baseline (i.e., RetinaNet)
```Shell
./tools/eval_retinanet_r50_400_050x.sh
```
- To evaluate FSAF (w/o anchor-based (AB))
```Shell
./tools/eval_fsaf_r50_400_050x.sh
```


## Benchmark

Below is benchmark results. All models are trained with an image-size of 400 and reduced LR-schedule for efficient experiments. Reproduced results show a similar aspect to the original paper (Table 1,2), demonstrating sanity of the implementation.

|  model        |    backbone    | img-size | LR-schd | box AP | box AP_50 | box AP_75 | download |
|:----------:   |:-------------: | :-----:  | :-----: | :----: | :------:  | :------:  | :------: |
| RetinaNet     |    R-50-FPN    |   400    |  0.50x  |  26.0  |   43.4    |   27.1    |  [model](https://drive.google.com/open?id=1rgjfNxMAicqrcX1-aB2xd-pYYUgigG8v) |
| FSAF (w/o AB) |    R-50-FPN    |   400    |  0.50x  |  26.2  |   44.7    |   26.5    |  [model](https://drive.google.com/open?id=153Rq7Q9hPQ_f7ntEa2hP1L1mXrO61Kok) |


## TODO
- Code reorganization is needed to be consistent with the style of MMDetection framework (current code is only written for fast prototyping). 


## Contact

- Ho-Deok Jang
- Email: hdjang@kaist.ac.kr
- Homepage: https://sites.google.com/view/hdjangcv