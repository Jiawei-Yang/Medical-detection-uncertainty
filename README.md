# Exploring Instance Level Uncertainty for Bounding-Box-Based Medical Detection
![Ovreall Arthetecture](https://github.com/Jiawei-Yang/Exploring-Instance-Level-Uncertainty-for-Bounding-Box-Based-Medical-Detection/blob/main/overview.png)

This repository contains PyTorch implementation of a medical detection model that enables the end-to-end estimation of bounding-box-level uncertainty. More details on method and implementation is described in a paper under review for ISBI2021.

## Installation Guide
Our implementation only depends on `Pytorch` and `XXX`. 

### Step 1: Clone the repository
```
$ git clone https://github.com/liangyuandg/DLCariesScreen.git
```

### Step 2: Install dependencies
**Note**: The implementation is built and tested under `PythonXXX`.

**Note**: If you are using a Python virtualenv, make sure it is activated before running each command in this guide.

Install PyTorch by following the [official guidance](https://pytorch.org/). 

Install XXX, which is a XXX library designed for computer vision research, by following the [official guidance](https://XXX).


## Model Overview

(a) The backbone is a multi-level single-scale Feature Pyramid Network (FPN) with levels P2, P3, P4, P5. 

(b) During training, bounding box predictions of probability, predictive variance, and location parameters are trained directly against ground-truth. 

(c) During inference, MC samples of bounding box for each pyramid level are first in-place aggregated for MC variances, which further averaged with predictive variances as the uncertainty estimation.

More details about the backbone ConvNet can be found in our paper and [MedicalImageDetectionToolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit).


## Notes
1. Since the privacy issue and commercial interest, the trained models and training images are not released at the moment. Request for the data should be made to liangyuandg@ucla.edu and will be considered according to the purpose of usage. By following the instructions above, new models can be trained on any individual's repository of data. 
2. More details about training strategy, model architecture descriptions, and result discussion will be released in a future publication.

