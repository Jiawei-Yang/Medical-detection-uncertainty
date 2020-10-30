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

class FPN(nn.Module):
    def __init__(self, start_filts = 48, dropout=False):
        super(FPN, self).__init__()

        self.dropout = dropout # Monte Carlo Dropout
        self.start_filts = start_filts
        self.end_filts = start_filts * 4
        end_filts = self.end_filts
        
        self.n_blocks = [3, 4, 6, 3] # resnet50
        self.block = ResBlock
        self.block_expansion = 4

        # feed +/- n neighbouring slices into channel dimension.
        self.C1 = conv(7, start_filts, ks=7, stride=2, pad=3)
        start_filts_exp = start_filts * self.block_expansion

        C2_layers = []
        C2_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        C2_layers.append(self.block(start_filts, start_filts, conv=conv, stride=1, downsample=(start_filts, self.block_expansion, 1)))
        for i in range(1, self.n_blocks[0]):
            C2_layers.append(self.block(start_filts_exp, start_filts, conv=conv))
       

        C3_layers = []
        C3_layers.append(self.block(start_filts_exp, start_filts * 2, conv=conv, stride=2, downsample=(start_filts_exp, 2, 2)))
        for i in range(1, self.n_blocks[1]):
            C3_layers.append(self.block(start_filts_exp * 2, start_filts * 2, conv=conv))

        C4_layers = []
        C4_layers.append(self.block(
            start_filts_exp * 2, start_filts * 4, conv=conv, stride=2,   downsample=(start_filts_exp * 2, 2, 2)))
        for i in range(1, self.n_blocks[2]):
            C4_layers.append(self.block(start_filts_exp * 4, start_filts * 4, conv=conv))


        C5_layers = []
        C5_layers.append(self.block(
            start_filts_exp * 4, start_filts * 8, conv=conv, stride=2,  downsample=(start_filts_exp * 4, 2, 2)))
        for i in range(1, self.n_blocks[3]):
            C5_layers.append(self.block(start_filts_exp * 8, start_filts * 8, conv=conv))
       
        self.C2 = nn.Sequential(*C2_layers)
        self.C3 = nn.Sequential(*C3_layers)

        ###### Dropout Layers #########
        if self.dropout:
            self.C4 = nn.Sequential(*C4_layers, nn.Dropout2D(p=0.1))
            self.C5 = nn.Sequential(*C5_layers, nn.Dropout2D(p=0.1))
        else:
            self.C4 = nn.Sequential(*C4_layers)
            self.C5 = nn.Sequential(*C5_layers)
        ###############################


        self.P1_upsample = Interpolate(scale_factor=2, mode='bilinear')
        self.P2_upsample = Interpolate(scale_factor=2, mode='bilinear')
        
        self.P5_conv1 = conv(start_filts*32, end_filts, ks=1, stride=1, relu=None)
        self.P4_conv1 = conv(start_filts*16, end_filts, ks=1, stride=1, relu=None)
        self.P3_conv1 = conv(start_filts*8, end_filts, ks=1, stride=1, relu=None)
        self.P2_conv1 = conv(start_filts*4, end_filts, ks=1, stride=1, relu=None)
        self.P1_conv1 = conv(start_filts, end_filts, ks=1, stride=1, relu=None)

        self.P1_conv2 = conv(end_filts, end_filts, ks=3, stride=1, pad=1, relu=None)
        self.P2_conv2 = conv(end_filts, end_filts, ks=3, stride=1, pad=1, relu=None)
        self.P3_conv2 = conv(end_filts, end_filts, ks=3, stride=1, pad=1, relu=None)

        ###### Dropout Layers #########
        if self.dropout:
            self.P4_conv2 = nn.Sequential(conv(192, 192, ks=3, stride=1, pad=1, relu=None), nn.Dropout2D(p=0.1))
            self.P5_conv2 = nn.Sequential(conv(192, 192, ks=3, stride=1, pad=1, relu=None), nn.Dropout2D(p=0.1))
        else:
            self.P4_conv2 = nn.Sequential(conv(192, 192, ks=3, stride=1, pad=1, relu=None))
            self.P5_conv2 = nn.Sequential(conv(192, 192, ks=3, stride=1, pad=1, relu=None))
        ###############################
        
    def forward(self, x):

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
        self.conv1 = conv(start_filts, planes, ks=1, stride=stride, norm=norm, relu=relu)
        self.conv2 = conv(planes, planes, ks=3, pad=1, norm=norm, relu=relu)
        self.conv3 = conv(planes, planes * 4, ks=1, norm=norm, relu=None)
        self.relu = nn.ReLU(inplace=True) if relu == 'relu' else nn.LeakyReLU(inplace=True)
        if downsample is not None:
            self.downsample = conv(downsample[0], downsample[0] * downsample[1], ks=1, stride=downsample[2], norm=norm, relu=None)
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