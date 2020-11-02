#!/usr/bin/env python
# Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import numpy as np
import torch.nn as nn


class configs:

    def __init__(self, server_env=None):

        #########################
        #    Preprocessing      #
        #########################
        self.root_dir = '/dfsdata2/houfeng1_data/yjw/Dataset/luna16'
        self.raw_data_dir = '{}/data'.format(self.root_dir)
        # training data annotation helper
        self.seg_dir = '{}/seg'.format(self.root_dir)
        self.pp_dir = '{}/pp_norm'.format(self.root_dir)
        self.target_spacing = (0.7, 0.7, 1.25)

        #########################
        #         I/O           #
        #########################
        self.model = 'MNet'
        self.backbone_path = ''
        self.model_path = ''

        # path to preprocessed data.
        self.pp_name = 'pp_norm'
        self.input_df_name = 'sorted_info_df.pickle'
        self.pp_data_path = '/dfsdata2/houfeng1_data/yjw/Dataset/luna16/{}'.format(self.pp_name)
        self.pp_test_data_path = self.pp_data_path #change if test_data in separate folder.

    
        #########################
        #      Data Loader      #
        #########################
        # select modalities from preprocessed data
        self.n_workers = 16
        # patch_size to be used for training. pre_crop_size is the patch_size before data augmentation.
        self.pre_crop_size = [300, 300]
        self.patch_size = [288, 288]

        # ratio of free sampled batch elements before class balancing is triggered
        # (>0 to include "empty"/background patches.)
        self.batch_sample_slack = 0.2

        # set 2D network to operate in 3D images.
        self.merge_2D_to_3D_preds = True

        #########################
        #  Schedule / Selection #
        #########################
        self.weight_decay = 0
        self.num_epochs = 250
        self.num_train_batches = 200 # number of feedforwards in one epoch
        self.batch_size = 20

        self.do_validation = True
        # decide whether to validate on entire patient volumes (like testing) or sampled patches (like training)
        # the former is morge accurate, while the latter is faster (depending on volume size)
        self.val_mode = 'val_sampling' # one of 'val_sampling' , 'val_patient'
        self.num_val_batches = 50

        #########################
        #   Testing / Plotting  #
        #########################
        # set the top-n-epochs to be saved for temporal averaging in testing.
        self.save_n_models = 10
        self.test_n_epochs = 1
        # set a minimum epoch number for saving in case of instabilities in the first phase of training.
        self.min_save_thresh = 0

        self.report_score_level = ['rois']  # choose list from 'patient', 'rois'
        self.class_dict = {1: 'nodule'}  # 0 is background.
        self.patient_class_of_interest = 1  # patient metrics are only plotted for one class.
        self.ap_match_ious = [0.5]  # list of ious to be evaluated for ap-scoring.

        self.model_selection_criteria = ['nodule_ap'] # criteria to average over for saving epochs.
        self.min_det_thresh = 0.1  # minimum confidence value to select predictions for evaluation.

        # threshold for clustering predictions together (wcs = weighted cluster scoring).
        # needs to be >= the expected overlap of predictions coming from one model (typically NMS threshold).
        # if too high, preds of the same object are separate clusters.
        self.wcs_iou = 1e-5

        self.plot_prediction_histograms = False
        self.plot_stat_curves = False

        #########################
        #   Data Augmentation   #
        #########################
        self.da_kwargs={
        'do_elastic_deform': True,
        'alpha':(0., 1500.),
        'sigma':(30., 50.),
        'do_rotation':True,
        'angle_x': (0., 2 * np.pi),
        'angle_y': (0., 0),
        'angle_z': (0., 0),
        'do_scale': True,
        'scale':(0.8, 1.1),
        'random_crop':False,
        'rand_crop_dist':  (self.patch_size[0] / 2. - 3, self.patch_size[1] / 2. - 3),
        'border_mode_data': 'constant',
        'border_cval_data': 0,
        'order_data': 1
        }

      
        #########################
        #   Add model specifics #
        #########################
        # learning rate is a list with one entry per epoch.
        self.learning_rate = [1e-4] * self.num_epochs
        # number of classes for head networks: n_foreground_classes + 1 (background)
        self.head_classes = 2

        # feature map strides per pyramid level are inferred from architecture.
        self.backbone_strides = {'xy': [4, 8, 16, 32]}

        # anchor scales are chosen according to expected object sizes in data set. Default uses only one anchor scale
        # per pyramid level. (outer list are pyramid levels (corresponding to BACKBONE_STRIDES), inner list are scales per level.)
        self.rpn_anchor_scales = {'xy': [[8], [16], [32], [64]]}

        # choose which pyramid levels to extract features from: P2: 0, P3: 1, P4: 2, P5: 3.
        self.pyramid_levels = [0, 1, 2, 3]

        # number of feature maps in rpn. typically lowered in 3D to save gpu-memory.
        self.n_rpn_features = 512 

        # anchor ratios and strides per position in feature maps.
        self.rpn_anchor_ratios = [1]
        self.rpn_anchor_stride = 1

        # Threshold for first stage (RPN) non-maximum suppression (NMS):  LOWER == HARDER SELECTION
        self.rpn_nms_threshold = 0.7 

        # loss sampling settings.
        self.rpn_train_anchors_per_image = 6  #per batch element
        self.train_rois_per_image = 6 #per batch element
        self.roi_positive_ratio = 0.5
        self.anchor_matching_iou = 0.7

        # factor of top-k candidates to draw from  per negative sample (stochastic-hard-example-mining).
        # poolsize to draw top-k candidates from will be shem_poolsize * n_negative_samples.
        self.shem_poolsize = 10

        self.rpn_bbox_std_dev = np.array([0.1, 0.1, 0.2, 0.2])
        self.bbox_std_dev = np.array([0.1, 0.1, 0.2, 0.2])
        self.window = np.array([0, 0, self.patch_size[0], self.patch_size[1]])
        self.scale = np.array([self.patch_size[0], self.patch_size[1], self.patch_size[0], self.patch_size[1]])

        # pre-selection in proposal-layer (stage 1) for NMS-speedup. applied per batch element.
        self.pre_nms_limit = 3000


        # Final selection of detections (refine_detections)
        self.model_max_instances_per_batch_element = 10  # per batch element and class.
        self.detection_nms_threshold = 1e-5  # needs to be > 0, otherwise all predictions are one cluster.
        self.model_min_confidence = 0.1

        self.backbone_shapes = np.array(
                [[int(np.ceil(self.patch_size[0] / stride)),
                  int(np.ceil(self.patch_size[1] / stride))]
                 for stride in self.backbone_strides['xy']])

        self.n_anchors_per_pos = 1

        self.n_rpn_features = 256

        # pre-selection of detections for NMS-speedup. per entire batch.
        self.pre_nms_limit = 10000

        # anchor matching iou is lower than in Mask R-CNN according to https://arxiv.org/abs/1708.02002
        self.anchor_matching_iou = 0.5


