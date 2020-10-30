# Exploring-Instance-Level-Uncertainty-for-Bounding-Box-Based-Medical-Detection

This is the repository for the ISBI 2021 submission ``Exploring Instance-Level Uncertainty for Bounding-Box-Based Medical Detection``

The models are heaviliy adopted and modified from [Jaeger et al.](https://github.com/MIC-DKFZ/medicaldetectiontoolkit).


![alt text](https://github.com/Jiawei-Yang/Exploring-Instance-Level-Uncertainty-for-Bounding-Box-Based-Medical-Detection/blob/main/overview.png)

(a) The backbone is a multi-level single-scale Feature Pyramid Network (FPN) with levels P2, P3, P4, P5. 
(b) During training, bounding box predictions of probability, predictive variance, and location parameters are trained directly against ground-truth. 
(c) During inference, MC samples of bounding box for each pyramid level are first in-place aggregated for MC variances, which further averaged with predictive variances as the uncertainty estimation.



Details about the network can be found in our paper and [MedicalImageDetectionToolkit](https://github.com/MIC-DKFZ/medicaldetectiontoolkit).

