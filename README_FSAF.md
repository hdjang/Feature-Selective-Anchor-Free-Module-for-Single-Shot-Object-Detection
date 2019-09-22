## Description

This repository reproduces the "Feature Selective Anchor-Free Module for Single-Shot Object Detection" (FSAF) [arXiv](https://arxiv.org/pdf/1903.00621.pdf). The implementation is based on MMDetection framework. All the extra codes that are added to the baseline (i.e., RetinaNet) is written according to the original paper.


## Get Started

Please follow [GETTING_STARTED.md](docs/GETTING_STARTED.md) of MMDetection to use this repo.


## Train/Eval

- To train baseline (i.e., RetinaNet)
```Shell
./tools/dist_train_retina_exp1_050x.sh
./tools/dist_train_retina_exp1_075x.sh
```
- To train FSAF implementation
```Shell
./tools/dist_train_fsaf_exp1_050x.sh
```

- To evaluate baseline (i.e., RetinaNet)
```Shell
./tools/eval_retina_exp1_050x.sh
./tools/eval_retina_exp1_075x.sh
```
- To evaluate FSAF implementation
```Shell
./tools/eval_fsaf_exp1_050x.sh
```


## Benchmark

Below is the evaluation results. We train all models with small img-size (i.e., 400) and reduced LR schedule for efficient experiments.

|  model     |    backbone    | img-size | LR-schd | box AP | download |
|:----------:|:-------------: | :-----:  | :-----: | :----: | :------: |
| RetinaNet  |    R-50-FPN    |   400    |  0.75x  |  29.7  |  [model](https:??.pth)  |
| RetinaNet  |    R-50-FPN    |   400    |  0.50x  |  27.9  |  [model](https:??.pth)  |
|:----------:|:-------------: | :-----:  | :-----: | :----: | :------: |
| FSAF       |    R-50-FPN    |   400    |  0.50x  |  -     |  [model](https:??.pth)  |


## Contact

Ho-Deok Jang
Email: hdjang@kaist.ac.kr
Homepage: https://sites.google.com/view/hdjangcv