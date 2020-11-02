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

import torch.nn as nn
import torch.nn.functional as F
import torch


def conv(c_in, c_out, ks, pad=0, stride=1, norm='batch_norm', relu='relu'):
    conv = nn.Conv2d(c_in, c_out, kernel_size=ks, padding=pad, stride=stride)
    if norm is not None:
        if norm == 'instance_norm':
            norm_layer = nn.InstanceNorm2d(c_out)
        else:
            norm_layer = nn.BatchNorm2d(c_out)
        conv = nn.Sequential(conv, norm_layer)
    if relu is not None:
        if relu == 'relu':
            relu_layer = nn.ReLU(inplace=True)
        elif relu == 'leaky_relu':
            relu_layer = nn.LeakyReLU(inplace=True)
        else:
            raise ValueError('relu type as specified in configs is not implemented...')
        conv = nn.Sequential(conv, relu_layer)



class FPN(nn.Module):
    """
    Feature Pyramid Network from https://arxiv.org/pdf/1612.03144.pdf with options for modifications.
    by default is constructed with Pyramid levels P2, P3, P4, P5.
    """
    def __init__(self, 
                input_channels = 7, # (3*2+1)feed +/- 3 neighbouring slices into channel dimension.
                start_filts = 48,
                out_channels = 192,
                res_architecture = 'resnet50', 
                dropout=False):
        """
        from configs:
        :param input_channels: number of channel dimensions in input data.
        :param start_filts:  number of feature_maps in first layer. rest is scaled accordingly.
        :param out_channels: number of feature_maps for output_layers of all levels in decoder.
        :param conv: instance of custom conv class containing the dimension info.
        :param res_architecture: string deciding whether to use "resnet50" or "resnet101".
        :param norm: string specifying type of feature map normalization. If None, no normalization is applied.
        :param relu: string specifying type of nonlinearity. If None, no nonlinearity is applied.
        """
        super(FPN, self).__init__()

        self.input_channels = input_channels
        self.start_filts = start_filts
        self.out_channels = out_channels
        self.n_blocks = [3, 4, {"resnet50": 6, "resnet101": 23}[res_architecture], 3]
        self.block = ResBlock
        self.block_expansion = 4
        self.dropout = dropout
        self.build_network()

    def build_network(self):

        self.C1 = conv(self.input_channels, self.start_filts, ks=7, stride=2, pad=3, norm=self.norm)
        start_filts_exp = self.start_filts * self.block_expansion


        C2_layers = []
        C2_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        C2_layers.append(self.block(self.start_filts, self.start_filts, conv=conv, stride=1, 
                         downsample=(self.start_filts, self.block_expansion, 1)))
        for i in range(1, self.n_blocks[0]):
            C2_layers.append(self.block(start_filts_exp, self.start_filts, conv=conv))

        C3_layers = []
        C3_layers.append(self.block(start_filts_exp, self.start_filts * 2, conv=conv, stride=2, 
                         downsample=(start_filts_exp, 2, 2)))
        for i in range(1, self.n_blocks[1]):
            C3_layers.append(self.block(start_filts_exp * 2, self.start_filts * 2, conv=conv))
        

        C4_layers = []
        C4_layers.append(self.block(
            start_filts_exp * 2, self.start_filts * 4, conv=conv, stride=2,  downsample=(start_filts_exp * 2, 2, 2)))
        for i in range(1, self.n_blocks[2]):
            C4_layers.append(self.block(start_filts_exp * 4, self.start_filts * 4, conv=conv))
        

        C5_layers = []
        C5_layers.append(self.block(
            start_filts_exp * 4, self.start_filts * 8, conv=conv, stride=2,  downsample=(start_filts_exp * 4, 2, 2)))
        for i in range(1, self.n_blocks[3]):
            C5_layers.append(self.block(start_filts_exp * 8, self.start_filts * 8, conv=conv))


        self.C2 = nn.Sequential(*C2_layers)
        self.C3 = nn.Sequential(*C3_layers)
        #########################
        #         Dropout       #
        #########################
        if self.dropout:
            self.C4 = nn.Sequential(*C4_layers, nn.Dropout2D(p=0.1))
            self.C5 = nn.Sequential(*C5_layers, nn.Dropout2D(p=0.1))
        else:
            self.C4 = nn.Sequential(*C4_layers)
            self.C5 = nn.Sequential(*C5_layers)
        ##########################

        self.P1_upsample = Interpolate(scale_factor=2, mode='bilinear')
        self.P2_upsample = Interpolate(scale_factor=2, mode='bilinear')
       

        self.P5_conv1 = conv(self.start_filts*32, self.out_channels, ks=1, stride=1, relu=None) 
        self.P4_conv1 = conv(self.start_filts*16, self.out_channels, ks=1, stride=1, relu=None)
        self.P3_conv1 = conv(self.start_filts*8, self.out_channels, ks=1, stride=1, relu=None)
        self.P2_conv1 = conv(self.start_filts*4, self.out_channels, ks=1, stride=1, relu=None)
        self.P1_conv1 = conv(self.start_filts, self.out_channels, ks=1, stride=1, relu=None)


        self.P1_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
        self.P2_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
        self.P3_conv2 = conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
        #########################
        #         Dropout       #
        #########################
        if self.dropout:
            self.P4_conv2 = nn.Sequential(conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None), 
                                          nn.Dropout2D(p=0.1))
            self.P5_conv2 = nn.Sequential(conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None), 
                                          nn.Dropout2D(p=0.1))
        else:
            self.P4_conv2 = nn.Sequential(conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None))
            self.P5_conv2 = nn.Sequential(conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None))
        ##########################



    def forward(self, x):
        """
        :param x: input image of shape (b, c, y, x, (z))
        :return: list of output feature maps per pyramid level, each with shape (b, c, y, x, (z)).
        """
        c0_out = x
        c1_out = self.C1(c0_out)
        c2_out = self.C2(c1_out)
        c3_out = self.C3(c2_out)
        c4_out = self.C4(c3_out)
        c5_out = self.C5(c4_out)
        p5_pre_out = self.P5_conv1(c5_out)

        p4_pre_out = self.P4_conv1(c4_out) + F.interpolate(p5_pre_out, scale_factor=2)
        p3_pre_out = self.P3_conv1(c3_out) + F.interpolate(p4_pre_out, scale_factor=2)
        p2_pre_out = self.P2_conv1(c2_out) + F.interpolate(p3_pre_out, scale_factor=2)

        p2_out = self.P2_conv2(p2_pre_out)
        p3_out = self.P3_conv2(p3_pre_out)
        p4_out = self.P4_conv2(p4_pre_out)
        p5_out = self.P5_conv2(p5_pre_out)
        out_list = [p2_out, p3_out, p4_out, p5_out]

        return out_list



class ResBlock(nn.Module):

    def __init__(self, start_filts, planes, conv, stride=1, downsample=None, norm=None, relu='relu'):
        super(ResBlock, self).__init__()
        self.conv1 = conv(self.start_filts, planes, ks=1, stride=stride, relu=relu)
        self.conv2 = conv(planes, planes, ks=3, pad=1, relu=relu)
        self.conv3 = conv(planes, planes * 4, ks=1, relu=None)
        self.relu = nn.ReLU(inplace=True) if relu == 'relu' else nn.LeakyReLU(inplace=True)
        if downsample is not None:
            self.downsample = conv(downsample[0], downsample[0] * downsample[1], ks=1, stride=downsample[2], relu=None)
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x