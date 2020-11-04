# Exploring Instance Level Uncertainty for Bounding-Box-Based Medical Detection
![Ovreall Arthetecture](https://github.com/Jiawei-Yang/Exploring-Instance-Level-Uncertainty-for-Bounding-Box-Based-Medical-Detection/blob/main/overview.png)

This repository contains PyTorch implementation of a medical detection model that enables the end-to-end estimation of bounding-box-level uncertainty. More details on method and implementation are included in a paper currently under review for ISBI2021.

## Installation Guide
Our implementation depends on `Pytorch` and [`MedicalImageDetectionToolkit`](https://github.com/MIC-DKFZ/medicaldetectiontoolkit). 

### Step 1: Clone the repository
```
$ git clone https://github.com/Jiawei-Yang/Exploring-Instance-Level-Uncertainty-for-Bounding-Box-Based-Medical-Detection.git
```

### Step 2: Install dependencies
**Note**: The implementation is built and tested under `Python3.6`.

**Note**: If you are using a Python virtualenv, make sure it is activated before running each command in this guide.

Install PyTorch0.4 by following the [official guidance](https://pytorch.org/). 

Compile Non-Maximum Suppression cuda function:
```
cd cuda_functions/nms_xD/src/cuda/
nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=[arch]
cd ../../
python build.py
cd ../
```
### Step 3: Run inference and training 

1. Set I/O paths and training specifics in the configs file: configs.py

2. Train the model with different modes: use ``-mc_var`` to enable  MC variances and ``-pred_var`` to enable predictive variances. 

For example:
```
python exec.py --mode train -mc_var -pred_var
```
enables both variances.

3. Run inference with different modes:
For example:
```
python exec.py --mode test -mc_var -pred_var
```

## Model Overview

(a) The model contains a multi-level single-scale Feature Pyramid Network (FPN) as the base detector.  

(b) The bounding box probability, predictive variance, and box location parameters are the output of our model. Those values are trained directly against ground-truth. 

(c) During inference, MC samples of bounding box for each pyramid level are first in-place aggregated for MC variances. The resulted MC variances are further averaged with predictive variances as the uncertainty estimation.

(d) Post-processing of Weighted Box Clustering (WBC) is utilized to reduce overlapping box predictions, which is similar to NMS and can be found in [Jaeger's](https://arxiv.org/abs/1811.08661).  

The base detector implementation is adapted from an existing work that achieves state-of-the-art detection accuracy. More details about it can be found in [MedicalImageDetectionToolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit).


## Notes
1. The method of enabling instance-level uncertainty can be applied to any ConvNet as the base detecotr. 
2. More details about training strategy, model architecture descriptions, and result discussion will be released in a future publication.

