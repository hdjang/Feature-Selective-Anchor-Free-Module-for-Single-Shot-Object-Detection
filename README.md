## Description

This repository implements "Zhu et al. Feature Selective Anchor-Free Module for Single-Shot Object Detection. CVPR, 2019." (FSAF) [PDF](https://arxiv.org/pdf/1903.00621.pdf). The implementation is based on MMDetection framework. All the codes for the FSAF model follow the original paper.


## Get Started

To use this repo, please follow [README.md](./README_MMDetection.md) of MMDetection.


## Train/Eval

**Train**
- To train baseline (i.e., RetinaNet)
```Shell
./tools/dist_train_retinanet_r50_400_075x.sh
./tools/dist_train_retinanet_r50_400_050x.sh
```
- To train FSAF implementation
```Shell
./tools/dist_train_fsaf_r50_400_050x.sh
```
**Eval**
For evaluation, pretrained model-weights should be located at "./models/here".

- To evaluate baseline (i.e., RetinaNet)
```Shell
./tools/eval_retinanet_r50_400_075x.sh
./tools/eval_retinanet_r50_400_050x.sh
```
- To evaluate FSAF implementation
```Shell
./tools/eval_fsaf_r50_400_050x.sh
```


## Benchmark

Below is the benchmark results. We train all models with an img-size of 400 and reduced LR-schedule for efficient experiments.

Currently, for the FSAF, this repo only provides train/eval codes and does not provide benchmark result and pretrained model due to slow training speed of the model.

|  model     |    backbone    | img-size | LR-schd | box AP | download |
|:----------:|:-------------: | :-----:  | :-----: | :----: | :------: |
| RetinaNet  |    R-50-FPN    |   400    |  0.75x  |  29.7  |  [model](https://drive.google.com/open?id=1AQYh1vVhPF8w8U_rt_iaHbHXhli7A_gi)  |
| RetinaNet  |    R-50-FPN    |   400    |  0.50x  |  27.9  |  [model](https://drive.google.com/open?id=1cijBcaLAtwqkrNmtgaLw6-VOByZ5pQTs)  |
| FSAF       |    R-50-FPN    |   400    |  0.50x  |   -    |    -     |


## Contact

- Ho-Deok Jang
- Email: hdjang@kaist.ac.kr
- Homepage: https://sites.google.com/view/hdjangcv