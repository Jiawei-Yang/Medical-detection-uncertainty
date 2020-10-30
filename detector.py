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



import utils.model_utils as mutils
import utils.exp_utils as utils
import sys
sys.path.append('../')
from cuda_functions.nms_2D.pth_nms import nms_gpu as nms_2D
from cuda_functions.nms_3D.pth_nms import nms_gpu as nms_3D

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from experiments.luna_exp import backbone

def conv(c_in, c_out, ks, pad=0, stride=1, norm=None, relu='relu'):
    conv = nn.Conv2d(c_in, c_out, kernel_size=ks, padding=pad, stride=stride)
    if norm is not None:
        if norm == 'instance_norm':
            norm_layer = nn.InstanceNorm2d(c_out)
        else:
            norm_layer = nn.BatchNorm2d(c_out)
        conv = nn.Sequential(conv, norm_layer)

############################################################
#  Network Heads
############################################################

class Classifier(nn.Module):


    def __init__(self, end_filts=192, n_rpn_features=256, pred_var=False):
        """
        Builds the classifier sub-network.
        """
        self.pred_var = pred_var

        super(Classifier, self).__init__()
        self.n_classes = 2 # nodule v.s. non-nodule
        n_input_channels = end_filts
        n_features = n_rpn_features

        self.conv_1 = conv(n_input_channels, n_features, ks=3, stride=1, pad=1)
        self.conv_2 = conv(n_features, n_features, ks=3, stride=1, pad=1)
        self.conv_3 = conv(n_features, n_features, ks=3, stride=1, pad=1)
        self.conv_4 = conv(n_features, n_features, ks=3, stride=1, pad=1)
        self.conv_final = conv(n_features, self.n_classes, ks=3, stride=1, pad=1, relu=None)

        #######################################################################
        if self.pred_var:
            self.conv_var_pred = conv(n_features, self.n_classes, ks=3, stride=1, pad=1, relu=None)
        #######################################################################

    def forward(self, x):
        """
        :param x: input feature map (b, in_c, y, x, (z))
        :return: class_logits (b, n_anchors, n_classes)
        """
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        class_logits = self.conv_final(x)
        axes = (0, 2, 3, 1) 
        class_logits = class_logits.permute(*axes)
        class_logits = class_logits.contiguous()
        class_logits = class_logits.view(x.size()[0], -1, self.n_classes)
        
        #######################################################################
        if self.pred_var:
            class_logits_var = self.conv_var_pred(x)
            class_logits_var = class_logits_var.permute(*axes)
            class_logits_var = class_logits_var.contiguous()
            class_logits_var = class_logits_var.view(x.size()[0], -1, self.n_classes)
        
            return [class_logits], [class_logits_var]
        else:
            return [class_logits]
        #######################################################################


class BBRegressor(nn.Module):


    def __init__(self, conv, end_filts=192, n_rpn_features=256):
        """
        Builds the bb-regression sub-network.
        """
        super(BBRegressor, self).__init__()
        n_input_channels = end_filts
        n_features = n_rpn_features

        self.conv_1 = conv(n_input_channels, n_features, ks=3, stride=1, pad=1)
        self.conv_2 = conv(n_features, n_features, ks=3, stride=1, pad=1)
        self.conv_3 = conv(n_features, n_features, ks=3, stride=1, pad=1)
        self.conv_4 = conv(n_features, n_features, ks=3, stride=1, pad=1)
        self.conv_final = conv(n_features, 4, ks=3, stride=1, pad=1, relu=None)


    def forward(self, x):
        """
        :param x: input feature map (b, in_c, y, x, (z))
        :return: bb_logits (b, n_anchors, dim * 2)
        """
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        bb_logits = self.conv_final(x)

        axes = (0, 2, 3, 1)
        bb_logits = bb_logits.permute(*axes)
        bb_logits = bb_logits.contiguous()
        bb_logits = bb_logits.view(x.size()[0], -1, 4) # coordinates deltas

        return [bb_logits]


############################################################
#  Loss Functions
############################################################

def compute_class_loss(anchor_matches, class_pred_logits, shem_poolsize=20, pred_var_logits=None, n_mcs=10):
    """
    :param anchor_matches: (n_anchors). [-1, 0, class_id] for negative, neutral, and positive matched anchors.
    :param class_pred_logits: (n_anchors, n_classes). logits from classifier sub-network.
    :param shem_poolsize: int. factor of top-k candidates to draw from per negative sample (online-hard-example-mining).
    :return: loss: torch tensor.
    :return: np_neg_ix: 1D array containing indices of the neg_roi_logits, which have been sampled for training.
    """
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    pos_indices = torch.nonzero(anchor_matches > 0)
    neg_indices = torch.nonzero(anchor_matches == -1)

    if pred_var_logits is not None:
        mu_mc = class_pred_logits.unsqueeze(2)
        mu_mc = mu_mc.repeat([1,1,n_mcs])
        std = torch.exp(pred_var_logits)
        std = std.unsqueeze(2)
        std = std.repeat([1, 1, n_mcs])
        noise = torch.zeros_like(mu_mc).normal_()*std
        class_pred_logits = mu_mc + noise

        if 0 not in pos_indices.size():
            pos_indices = pos_indices.squeeze(1)
            roi_logits_pos = class_pred_logits[pos_indices]
            targets_pos = anchor_matches[pos_indices]
            targets_pos = targets_pos.unsqueeze(1)
            targets_pos = targets_pos.repeat([1,n_mcs])
            pos_loss = F.cross_entropy(roi_logits_pos, targets_pos.long())
        else:
            pos_loss = torch.FloatTensor([0]).cuda()
        
        # get negative samples, such that the amount matches the number of positive samples, but at least 1.
        # get high scoring negatives by applying online-hard-example-mining.
        if 0 not in neg_indices.size():
            neg_indices = neg_indices.squeeze(1)
            roi_logits_neg = class_pred_logits[neg_indices]
            negative_count = np.max((1, pos_indices.size()[0]))
            roi_probs_neg = F.softmax(roi_logits_neg, dim=1)
            roi_probs_neg = torch.mean(roi_probs_neg,dim=-1)
            neg_ix = mutils.shem(roi_probs_neg, negative_count, shem_poolsize)
            neg_pos = torch.LongTensor([0] * neg_ix.shape[0]).cuda()
            neg_pos = neg_pos.unsqueeze(1)
            neg_pos = neg_pos.repeat([1,n_mcs])
            neg_loss = F.cross_entropy(roi_logits_neg[neg_ix], neg_pos)
            np_neg_ix = neg_ix.cpu().data.numpy()
        else:
            neg_loss = torch.FloatTensor([0]).cuda()
            np_neg_ix = np.array([]).astype('int32')

        loss = (pos_loss + neg_loss) / 2
        return loss, np_neg_ix

    else:

        # get positive samples and calucalte loss.
        if 0 not in pos_indices.size():
            pos_indices = pos_indices.squeeze(1)
            roi_logits_pos = class_pred_logits[pos_indices]
            targets_pos = anchor_matches[pos_indices]
            pos_loss = F.cross_entropy(roi_logits_pos, targets_pos.long())
        else:
            pos_loss = torch.FloatTensor([0]).cuda()

        # get negative samples, such that the amount matches the number of positive samples, but at least 1.
        # get high scoring negatives by applying online-hard-example-mining.
        if 0 not in neg_indices.size():
            neg_indices = neg_indices.squeeze(1)
            roi_logits_neg = class_pred_logits[neg_indices]
            negative_count = np.max((1, pos_indices.size()[0]))
            roi_probs_neg = F.softmax(roi_logits_neg, dim=1)
            neg_ix = mutils.shem(roi_probs_neg, negative_count, shem_poolsize)
            neg_loss = F.cross_entropy(roi_logits_neg[neg_ix], torch.LongTensor([0] * neg_ix.shape[0]).cuda())
            # return the indices of negative samples, which contributed to the loss (for monitoring plots).
            np_neg_ix = neg_ix.cpu().data.numpy()
        else:
            neg_loss = torch.FloatTensor([0]).cuda()
            np_neg_ix = np.array([]).astype('int32')

        loss = (pos_loss + neg_loss) / 2
        return loss, np_neg_ix


def compute_bbox_loss(target_deltas, pred_deltas, anchor_matches):
    """
    :param target_deltas:   (b, n_positive_anchors, (dy, dx, (dz), log(dh), log(dw), (log(dd)))).
    Uses 0 padding to fill in unsed bbox deltas.
    :param pred_deltas: predicted deltas from bbox regression head. (b, n_anchors, (dy, dx, (dz), log(dh), log(dw), (log(dd))))
    :param anchor_matches: (n_anchors). [-1, 0, class_id] for negative, neutral, and positive matched anchors.
    :return: loss: torch 1D tensor.
    """
    if 0 not in torch.nonzero(anchor_matches > 0).size():

        indices = torch.nonzero(anchor_matches > 0).squeeze(1)
        # Pick bbox deltas that contribute to the loss
        pred_deltas = pred_deltas[indices]
        # Trim target bounding box deltas to the same length as pred_deltas.
        target_deltas = target_deltas[:pred_deltas.size()[0], :]
        # Smooth L1 loss
        loss = F.smooth_l1_loss(pred_deltas, target_deltas)
    else:
        loss = torch.FloatTensor([0]).cuda()

    return loss


############################################################
#  Output Handler
############################################################

def refine_detections(anchors, probs, deltas, batch_ixs, pred_var=None, mc_var=None):
    """
    Refine classified proposals, filter overlaps and return final
    detections. n_proposals here is typically a very large number: batch_size * n_anchors.
    This function is hence optimized on trimming down n_proposals.
    :param anchors: (n_anchors, 2 * dim)
    :param probs: (n_proposals, n_classes) softmax probabilities for all rois as predicted by classifier head.
    :param deltas: (n_proposals, n_classes, 2 * dim) box refinement deltas as predicted by bbox regressor head.
    :param batch_ixs: (n_proposals) batch element assignemnt info for re-allocation.
    :return: result: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score))
    """
    pre_nms_limit = 10000
    rpn_bbox_std_dev = np.array([0.1, 0.1, 0.1, 0.2])
    scale = np.array([288,288,288,288])
    window = np.array([0,0,288,288])

    anchors = anchors.repeat(len(np.unique(batch_ixs)), 1)

    # flatten foreground probabilities, sort and trim down to highest confidences by pre_nms limit.
    fg_probs = probs[:, 1:].contiguous()
    flat_probs, flat_probs_order = fg_probs.view(-1).sort(descending=True)
    keep_ix = flat_probs_order[:pre_nms_limit]
    # reshape indices to 2D index array with shape like fg_probs.
    keep_arr = torch.cat(((keep_ix / fg_probs.shape[1]).unsqueeze(1), (keep_ix % fg_probs.shape[1]).unsqueeze(1)), 1)

    pre_nms_scores = flat_probs[:pre_nms_limit]
    pre_nms_class_ids = keep_arr[:, 1] + 1  # add background again.
    pre_nms_batch_ixs = batch_ixs[keep_arr[:, 0]]
    pre_nms_anchors = anchors[keep_arr[:, 0]]
    pre_nms_deltas = deltas[keep_arr[:, 0]]
    keep = torch.arange(pre_nms_scores.size()[0]).long().cuda()
    

    # apply bounding box deltas. re-scale to image coordinates.
    std_dev = torch.from_numpy(np.reshape(rpn_bbox_std_dev, [1, 4])).float().cuda()
    scale = torch.from_numpy(scale).float().cuda()
    refined_rois = mutils.apply_box_deltas_2D(pre_nms_anchors / scale, pre_nms_deltas * std_dev) * scale 

    # round and cast to int since we're deadling with pixels now
    refined_rois = mutils.clip_to_window(window, refined_rois)
    pre_nms_rois = torch.round(refined_rois)
    for j, b in enumerate(mutils.unique1d(pre_nms_batch_ixs)):

        bixs = torch.nonzero(pre_nms_batch_ixs == b)[:, 0]
        bix_class_ids = pre_nms_class_ids[bixs]
        bix_rois = pre_nms_rois[bixs]
        bix_scores = pre_nms_scores[bixs]

        for i, class_id in enumerate(mutils.unique1d(bix_class_ids)):

            ixs = torch.nonzero(bix_class_ids == class_id)[:, 0]
            # nms expects boxes sorted by score.
            ix_rois = bix_rois[ixs]
            ix_scores = bix_scores[ixs]
            ix_scores, order = ix_scores.sort(descending=True)
            ix_rois = ix_rois[order, :]
            ix_scores = ix_scores
            
            class_keep = nms_2D(torch.cat((ix_rois, ix_scores.unsqueeze(1)), dim=1), 1e-5)

            # map indices back.
            class_keep = keep[bixs[ixs[order[class_keep]]]]
            # merge indices over classes for current batch element
            b_keep = class_keep if i == 0 else mutils.unique1d(torch.cat((b_keep, class_keep)))

        # only keep top-k boxes of current batch-element.
        top_ids = pre_nms_scores[b_keep].sort(descending=True)[1][:10]
        b_keep = b_keep[top_ids]
        # merge indices over batch elements.
        batch_keep = b_keep if j == 0 else mutils.unique1d(torch.cat((batch_keep, b_keep)))

    keep = batch_keep

    # arrange output.
    ##################################
    if mc_var is not None:
        mc_var = mc_var[:, 1:].contiguous()
        pre_nms_mc_var = mc_var[keep_arr[:, 0]].squeeze()

        if pred_var is not None:
            pred_var = pred_var[:, 1:].contiguous()
            pre_nms_pred_var = pred_var[keep_arr[:, 0]].squeeze()
            result = torch.cat((pre_nms_rois[keep],
                        pre_nms_batch_ixs[keep].unsqueeze(1).float(),
                        pre_nms_class_ids[keep].unsqueeze(1).float(),
                        pre_nms_scores[keep].unsqueeze(1),
                        pre_nms_mc_var[keep].unsqueeze(1),
                        pre_nms_pred_var[keep].unsqueeze(1)), dim=1)
        else:
            result = torch.cat((pre_nms_rois[keep],
                        pre_nms_batch_ixs[keep].unsqueeze(1).float(),
                        pre_nms_class_ids[keep].unsqueeze(1).float(),
                        pre_nms_scores[keep].unsqueeze(1),
                        pre_nms_mc_var[keep].unsqueeze(1)), dim=1)
    else:
        result = torch.cat((pre_nms_rois[keep],
                            pre_nms_batch_ixs[keep].unsqueeze(1).float(),
                            pre_nms_class_ids[keep].unsqueeze(1).float(),
                            pre_nms_scores[keep].unsqueeze(1)), dim=1)

    return result



def get_results(img_shape, detections, box_results_list=None, mc_var=False, pred_var=False):

    detections = detections.cpu().data.numpy()
    batch_ixs = detections[:, 4]
    detections = [detections[batch_ixs == ix] for ix in range(img_shape[0])]

    # for test_forward, where no previous list exists.
    if box_results_list is None:
        box_results_list = [[] for _ in range(img_shape[0])]

    for ix in range(img_shape[0]):
        if 0 not in detections[ix].shape:
            boxes = detections[ix][:, :4].astype(np.int32)
            class_ids = detections[ix][:, 5].astype(np.int32)
            scores = detections[ix][:, 6]
            if mc_var:
                mc_vars = detections[ix][:, 7]
                if pred_var:
                    pred_vars = detections[ix][:, 8]

            exclude_ix = np.where((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
            
            if exclude_ix.shape[0] > 0:
                boxes = np.delete(boxes, exclude_ix, axis=0)
                class_ids = np.delete(class_ids, exclude_ix, axis=0)
                scores = np.delete(scores, exclude_ix, axis=0)
                if mc_var:
                    mc_vars = np.delete(mc_vars, exclude_ix, axis=0)
                    if pred_var:
                        pred_vars = np.delete(pred_vars, exclude_ix, axis=0)

            if 0 not in boxes.shape:
                for ix2 in range(len(scores)):
                    if scores[ix2] >= 0.1:
                        if mc_var:
                            if pred_var:
                                box_results_list[ix].append({'box_coords': boxes[ix2],
                                                             'box_score': scores[ix2],
                                                             'mc_var': mc_vars[ix2],
                                                             'pred_var': pred_vars[ix2],
                                                             'box_type': 'det',
                                                             'box_pred_class_id': class_ids[ix2]})
                            else:
                                box_results_list[ix].append({'box_coords': boxes[ix2],
                                                             'box_score': scores[ix2],
                                                             'mc_var': mc_vars[ix2],
                                                             'box_type': 'det',
                                                             'box_pred_class_id': class_ids[ix2]})
                        else:
                            box_results_list[ix].append({'box_coords': boxes[ix2],
                                                         'box_score': scores[ix2],
                                                         'box_type': 'det',
                                                         'box_pred_class_id': class_ids[ix2]})

        results_dict = {'boxes': box_results_list}

    return results_dict



class net(nn.Module):


    def __init__(self, logger, mc_var=False, pred_var=False):

        super(net, self).__init__()
        self.pred_var = pred_var
        self.mc_var = False

        self.logger = logger
        self.build()
      
        mutils.initialize_weights(self)
       
    def build(self):
        """
        Build Retina Net architecture.
        """
        # Image size must be dividable by 2 multiple times.
        h, w = [288,288]
        if h / 2 ** 5 != int(h / 2 ** 5) or w / 2 ** 5 != int(w / 2 ** 5):
            raise Exception("Image size must be dividable by 2 at least 5 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # instanciate abstract multi dimensional conv class and backbone model.
        conv = mutils.NDConvGenerator(2)

        # build Anchors, FPN, Classifier / Bbox-Regressor -head
        self.np_anchors = mutils.generate_pyramid_anchors(self.logger)
        self.anchors = torch.from_numpy(self.np_anchors).float().cuda()
        self.Fpn = backbone.FPN(dropout=self.mc_var)
        self.Classifier = Classifier()
        self.BBRegressor = BBRegressor()


    def train_forward(self, batch, **kwargs):
        """
        train method (also used for validation monitoring). wrapper around forward pass of network. prepares input data
        for processing, computes losses, and stores outputs in a dictionary.
        :param batch: dictionary containing 'data', 'seg', etc.
        :return: results_dict: dictionary with keys:
                'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                        [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
                'seg_preds': pixelwise segmentation output (b, c, y, x, (z)) with values [0, .., n_classes].
                'monitor_values': dict of values to be monitored.
        """
        img = batch['data']
        gt_class_ids = batch['roi_labels']
        gt_boxes = batch['bb_target']

        img = torch.from_numpy(img).float().cuda()
        batch_class_loss = torch.FloatTensor([0]).cuda()
        batch_bbox_loss = torch.FloatTensor([0]).cuda()

        # list of output boxes for monitoring/plotting. each element is a list of boxes per batch element.
        box_results_list = [[] for _ in range(img.shape[0])]
        detections, class_logits, pred_deltas = self.forward(img)

        # loop over batch
        for b in range(img.shape[0]):

            # add gt boxes to results dict for monitoring.
            if len(gt_boxes[b]) > 0:
                for ix in range(len(gt_boxes[b])):
                    box_results_list[b].append({'box_coords': batch['bb_target'][b][ix],
                                                'box_label': batch['roi_labels'][b][ix], 'box_type': 'gt'})

                # match gt boxes with anchors to generate targets.
                anchor_class_match, anchor_target_deltas = mutils.gt_anchor_matching(
                    self.np_anchors, gt_boxes[b], gt_class_ids[b])

                # add positive anchors used for loss to results_dict for monitoring.
                pos_anchors = mutils.clip_boxes_numpy(
                    self.np_anchors[np.argwhere(anchor_class_match > 0)][:, 0], img.shape[2:])
                for p in pos_anchors:
                    box_results_list[b].append({'box_coords': p, 'box_type': 'pos_anchor'})

            else:
                anchor_class_match = np.array([-1]*self.np_anchors.shape[0])
                anchor_target_deltas = np.array([0])

            anchor_class_match = torch.from_numpy(anchor_class_match).cuda()
            anchor_target_deltas = torch.from_numpy(anchor_target_deltas).float().cuda()

            # compute losses.
            class_loss, neg_anchor_ix = compute_class_loss(anchor_class_match, class_logits[b])
            bbox_loss = compute_bbox_loss(anchor_target_deltas, pred_deltas[b], anchor_class_match)

            # add negative anchors used for loss to results_dict for monitoring.
            neg_anchors = mutils.clip_boxes_numpy(
                self.np_anchors[np.argwhere(anchor_class_match == -1)][0, neg_anchor_ix], img.shape[2:])
            for n in neg_anchors:
                box_results_list[b].append({'box_coords': n, 'box_type': 'neg_anchor'})

            batch_class_loss += class_loss / img.shape[0]
            batch_bbox_loss += bbox_loss / img.shape[0]

        results_dict = get_results(img.shape, detections, box_results_list, self.pred_var)
        loss = batch_class_loss + batch_bbox_loss
        results_dict['torch_loss'] = loss
        results_dict['monitor_values'] = {'loss': loss.item(), 'class_loss': batch_class_loss.item()}
        results_dict['logger_string'] = "loss: {0:.2f}, class: {1:.2f}, bbox: {2:.2f}"\
            .format(loss.item(), batch_class_loss.item(), batch_bbox_loss.item())

        return results_dict


    def test_forward(self, batch, **kwargs):
        """
        test method. wrapper around forward pass of network without usage of any ground truth information.
        prepares input data for processing and stores outputs in a dictionary.
        :param batch: dictionary containing 'data'
        :return: results_dict: dictionary with keys:
               'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                       [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
               'seg_preds': pixel-wise class predictions (b, 1, y, x, (z)) with values [0, ..., n_classes] for
                            retina_unet and dummy array for retina_net.
        """
        img = batch['data']
        img = torch.from_numpy(img).float().cuda()
        if self.mc_var:
            detections, _, _, mc_var= self.mc_test_forward(img)
        else:
            detections, _, _ = self.forward(img)
        results_dict = get_results(img.shape, detections, mc_var = mc_var)
        return results_dict


    def mc_test_forward(self, img):
        flat_softmax_list, bb_output_list, flat_var_list = []
        for t in range(10):
            fpn_outs = self.Fpn(img)
            class_layer_outputs, bb_reg_layer_outputs = [], []
            if self.pred_var:
                class_layer_vars = []
                for p in fpn_outs:
                    class_mu, class_var = self.Classifier(p)
                    bb_reg_layer_outputs.append(self.BBRegressor(p))
                    class_layer_outputs.append(class_mu)
                    class_layer_vars.append(class_var)
            else:
                for p in fpn_outs:
                    class_layer_outputs.append(self.Classifier(p))
                    bb_reg_layer_outputs.append(self.BBRegressor(p))

            class_logits = list(zip(*class_layer_outputs))
            class_logits = [torch.cat(list(o), dim=1) for o in class_logits][0]
            bb_outputs = list(zip(*bb_reg_layer_outputs))
            bb_outputs = [torch.cat(list(o), dim=1) for o in bb_outputs][0]

            batch_ixs = torch.arange(class_logits.shape[0]).unsqueeze(1).repeat(1, class_logits.shape[1]).view(-1).cuda()
            flat_class_softmax = F.softmax(class_logits.view(-1, class_logits.shape[-1]), 1)
            flat_bb_outputs = bb_outputs.view(-1, bb_outputs.shape[-1])

            flat_softmax_list.append(flat_class_softmax)
            bb_output_list.append(bb_outputs)

            if self.pred_var:
                class_var_logits = list(zip(*class_layer_vars)) 
                class_var_logits = [torch.cat(list(o), dim=1) for o in class_var_logits][0]
                flat_class_var = class_var_logits.view(-1, class_var_logits.shape[-1])
                flat_var_list.append(flat_class_var)

        flat_softmax_stacked = torch.stack(flat_softmax_list)
        flat_var_stacked = torch.stack(flat_var_list)
        bb_output_stacked= torch.stack(bb_output_list)

        flat_class_softmax = torch.mean(flat_softmax_stacked,axis=0)
        flat_bb_outputs = torch.mean(bb_output_stacked,axis=0)
        flat_class_var = torch.mean(flat_var_stacked,axis=0)
        
        mc_var = torch.var(flat_softmax_stacked,axis=0)
    
        
        if self.pred_var:  
            detections = refine_detections(self.anchors, flat_class_softmax, flat_bb_outputs, batch_ixs, pred_var=flat_class_var,mc_var=mc_var)
        else:
            detections = refine_detections(self.anchors, flat_class_softmax, flat_bb_outputs, batch_ixs, mc_var=mc_var)
        return detections, class_logits, bb_outputs



    def forward(self,img):
        # Feature extraction
        fpn_outs = self.Fpn(img)
        # Loop through pyramid layers
        class_layer_outputs, bb_reg_layer_outputs = [], []  # list of lists
        if self.pred_var:
            class_layer_vars = []
            for p in fpn_outs:
                class_mu, class_var = self.Classifier(p)
                bb_reg_layer_outputs.append(self.BBRegressor(p))
                class_layer_outputs.append(class_mu)
                class_layer_vars.append(class_var)
        else:
            for p in fpn_outs:
                class_layer_outputs.append(self.Classifier(p))
                bb_reg_layer_outputs.append(self.BBRegressor(p))

        class_logits = list(zip(*class_layer_outputs))
        class_logits = [torch.cat(list(o), dim=1) for o in class_logits][0]
        bb_outputs = list(zip(*bb_reg_layer_outputs))
        bb_outputs = [torch.cat(list(o), dim=1) for o in bb_outputs][0]

        # merge batch_dimension and store info in batch_ixs for re-allocation.
        batch_ixs = torch.arange(class_logits.shape[0]).unsqueeze(1).repeat(1, class_logits.shape[1]).view(-1).cuda()
        flat_class_softmax = F.softmax(class_logits.view(-1, class_logits.shape[-1]), 1)
        flat_bb_outputs = bb_outputs.view(-1, bb_outputs.shape[-1])

        if self.pred_var:
            class_var_logits = list(zip(*class_layer_vars)) 
            class_var_logits = [torch.cat(list(o), dim=1) for o in class_var_logits][0]
            flat_class_var = class_var_logits.view(-1, class_var_logits.shape[-1])

            detections = refine_detections(self.anchors, flat_class_softmax, flat_bb_outputs, batch_ixs, flat_class_var)
        else:
            detections = refine_detections(self.anchors, flat_class_softmax, flat_bb_outputs, batch_ixs)

        return detections, class_logits, bb_outputs
