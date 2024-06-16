
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F

from tractseg.libs.pytorch_utils import conv2d
from tractseg.libs.pytorch_utils import deconv2d


class UNet_Pytorch_Cnet(torch.nn.Module):
    def __init__(self, fix_other_layer, img_size, n_input_channels=3, n_classes=7, n_filt=64, batchnorm=False, dropout=False, upsample="bilinear"):
        super(UNet_Pytorch_Cnet, self).__init__()

        self.img_size = img_size
        self.use_dropout = dropout

        self.in_channel = n_input_channels
        self.n_classes = n_classes
        self.n_filt = n_filt

        ##------------------source_Fixed----------------------
        self.contr_1_1_source = conv2d(n_input_channels, n_filt, batchnorm=batchnorm)#1
        self.contr_1_2_source = conv2d(n_filt, n_filt, batchnorm=batchnorm)#2

        self.contr_2_1_source = conv2d(n_filt, n_filt * 2, batchnorm=batchnorm)#3
        self.contr_2_2_source = conv2d(n_filt * 2, n_filt * 2, batchnorm=batchnorm)#4

        self.contr_3_1_source = conv2d(n_filt * 2, n_filt * 4, batchnorm=batchnorm)#5
        self.contr_3_2_source = conv2d(n_filt * 4, n_filt * 4, batchnorm=batchnorm)#6

        self.contr_4_1_source = conv2d(n_filt * 4, n_filt * 8, batchnorm=batchnorm)#7
        self.contr_4_2_source = conv2d(n_filt * 8, n_filt * 8, batchnorm=batchnorm)#8

        self.encode_1_source = conv2d(n_filt * 8, n_filt * 16, batchnorm=batchnorm)#9
        self.encode_2_source = conv2d(n_filt * 16, n_filt * 16, batchnorm=batchnorm)#10

        self.deconv_1_source = deconv2d(n_filt * 16, n_filt * 16, kernel_size=2, stride=2)#11
        self.expand_1_1_source = conv2d(n_filt * 8 + n_filt * 16, n_filt * 8, batchnorm=batchnorm)#12
        self.expand_1_2_source = conv2d(n_filt * 8, n_filt * 8, batchnorm=batchnorm)#13

        self.deconv_2_source = deconv2d(n_filt * 8, n_filt * 8, kernel_size=2, stride=2)#14
        self.expand_2_1_source = conv2d(n_filt * 4 + n_filt * 8, n_filt * 4, stride=1, batchnorm=batchnorm)#15
        self.expand_2_2_source = conv2d(n_filt * 4, n_filt * 4, stride=1, batchnorm=batchnorm)#16

        self.deconv_3_source = deconv2d(n_filt * 4, n_filt * 4, kernel_size=2, stride=2)#17
        self.expand_3_1_source = conv2d(n_filt * 2 + n_filt * 4, n_filt * 2, stride=1, batchnorm=batchnorm)#18
        self.expand_3_2_source = conv2d(n_filt * 2, n_filt * 2, stride=1, batchnorm=batchnorm)#19

        self.deconv_4_source = deconv2d(n_filt * 2, n_filt * 2, kernel_size=2, stride=2)#20
        self.expand_4_1_source = conv2d(n_filt + n_filt * 2, n_filt, stride=1, batchnorm=batchnorm)#21
        self.expand_4_2_source = conv2d(n_filt, n_filt, stride=1, batchnorm=batchnorm)#22

        self.conv_5_new_source = nn.Conv2d(n_filt, n_classes, kernel_size=1, stride=1, padding=0, bias=True)#23

        self.pool_1 = nn.MaxPool2d((2, 2))
        self.pool_2 = nn.MaxPool2d((2, 2))
        self.pool_3 = nn.MaxPool2d((2, 2))
        self.pool_4 = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout(p=0.4)

        for p in self.parameters():
            p.requires_grad = False

        self.cnet_contr_1_1 = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Conv2d(n_filt,   n_filt*2,  kernel_size=1, stride=1, padding=0, bias=True))
        self.cnet_contr_1_2 = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Conv2d(n_filt,   n_filt*2,  kernel_size=1, stride=1, padding=0, bias=True))
        self.cnet_contr_2_1 = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Conv2d(n_filt*2, n_filt*4,  kernel_size=1, stride=1, padding=0, bias=True))
        self.cnet_contr_2_2 = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Conv2d(n_filt*2, n_filt*4,  kernel_size=1, stride=1, padding=0, bias=True))
        self.cnet_contr_3_1 = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Conv2d(n_filt*4, n_filt*8,  kernel_size=1, stride=1, padding=0, bias=True))
        self.cnet_contr_3_2 = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Conv2d(n_filt*4, n_filt*8,  kernel_size=1, stride=1, padding=0, bias=True)) 
        self.cnet_contr_4_1 = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Conv2d(n_filt*8, n_filt*16, kernel_size=1, stride=1, padding=0, bias=True))
        self.cnet_contr_4_2 = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Conv2d(n_filt*8, n_filt*16, kernel_size=1, stride=1, padding=0, bias=True)) 
       
        self.cnet_encode_1 = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Conv2d(n_filt*16, n_filt*32, kernel_size=1, stride=1, padding=0, bias=True))
        self.cnet_encode_2 = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Conv2d(n_filt*16, n_filt*32, kernel_size=1, stride=1, padding=0, bias=True)) 
        
        self.cnet_deconv_1  = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Conv2d(n_filt*16, n_filt*32, kernel_size=1, stride=1, padding=0, bias=True)) 
        self.cnet_expand_1_1= nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Conv2d(n_filt*8,  n_filt*16, kernel_size=1, stride=1, padding=0, bias=True)) 
        self.cnet_expand_1_2= nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Conv2d(n_filt*8,  n_filt*16, kernel_size=1, stride=1, padding=0, bias=True)) 
        self.cnet_deconv_2  = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Conv2d(n_filt*8,  n_filt*16, kernel_size=1, stride=1, padding=0, bias=True)) 
        self.cnet_expand_2_1= nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Conv2d(n_filt*4,  n_filt*8,  kernel_size=1, stride=1, padding=0, bias=True)) 
        self.cnet_expand_2_2= nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Conv2d(n_filt*4,  n_filt*8,  kernel_size=1, stride=1, padding=0, bias=True)) 
        self.cnet_deconv_3  = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Conv2d(n_filt*4,  n_filt*8,  kernel_size=1, stride=1, padding=0, bias=True)) 
        self.cnet_expand_3_1= nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Conv2d(n_filt*2,  n_filt*4,  kernel_size=1, stride=1, padding=0, bias=True)) 
        self.cnet_expand_3_2= nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Conv2d(n_filt*2,  n_filt*4,  kernel_size=1, stride=1, padding=0, bias=True)) 
        self.cnet_deconv_4  = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Conv2d(n_filt*2,  n_filt*4,  kernel_size=1, stride=1, padding=0, bias=True)) 
        self.cnet_expand_4_1= nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Conv2d(n_filt,    n_filt*2,  kernel_size=1, stride=1, padding=0, bias=True)) 
        self.cnet_expand_4_2= nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Conv2d(n_filt,    n_filt*2,  kernel_size=1, stride=1, padding=0, bias=True)) 

        self.cnet_conv_5= nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Conv2d(n_classes, n_classes*2, kernel_size=1, stride=1, padding=0, bias=True))


    def forward(self, input):
        ## get warmup pseudo prediction
        c_value_list = []

        contr_1_1 = self.contr_1_1_source(input)
        c_contr_1_1 = self.cnet_contr_1_1(contr_1_1)#[b,c*2,1,1]--logits
        c_value_list.append(c_contr_1_1)

        contr_1_2 = self.contr_1_2_source(contr_1_1)
        c_contr_1_2 = self.cnet_contr_1_2(contr_1_2)
        c_value_list.append(c_contr_1_2)

        pool_1 = self.pool_1(contr_1_2)

        contr_2_1 = self.contr_2_1_source(pool_1)
        c_contr_2_1 = self.cnet_contr_2_1(contr_2_1)
        c_value_list.append(c_contr_2_1)

        contr_2_2 = self.contr_2_2_source(contr_2_1)
        c_contr_2_2 = self.cnet_contr_2_2(contr_2_2)
        c_value_list.append(c_contr_2_2)

        pool_2 = self.pool_2(contr_2_2)

        contr_3_1 = self.contr_3_1_source(pool_2)
        c_contr_3_1 = self.cnet_contr_3_1(contr_3_1)
        c_value_list.append(c_contr_3_1)

        contr_3_2 = self.contr_3_2_source(contr_3_1)
        c_contr_3_2 = self.cnet_contr_3_2(contr_3_2)
        c_value_list.append(c_contr_3_2)

        pool_3 = self.pool_3(contr_3_2)

        contr_4_1 = self.contr_4_1_source(pool_3)
        c_contr_4_1 = self.cnet_contr_4_1(contr_4_1)
        c_value_list.append(c_contr_4_1)

        contr_4_2 = self.contr_4_2_source(contr_4_1)
        c_contr_4_2 = self.cnet_contr_4_2(contr_4_2)
        c_value_list.append(c_contr_4_2)

        pool_4 = self.pool_4(contr_4_2)

        if self.use_dropout:
            pool_4 = self.dropout(pool_4)

        encode_1 = self.encode_1_source(pool_4)
        c_encode_1 = self.cnet_encode_1(encode_1)
        c_value_list.append(c_encode_1)
        encode_2 = self.encode_2_source(encode_1)
        c_encode_2 = self.cnet_encode_2(encode_2)
        c_value_list.append(c_encode_2)
        deconv_1 = self.deconv_1_source(encode_2)
        c_deconv_1 = self.cnet_deconv_1(deconv_1)
        c_value_list.append(c_deconv_1) 

        concat1 = torch.cat([deconv_1, contr_4_2], 1)
        expand_1_1 = self.expand_1_1_source(concat1)
        c_expand_1_1 = self.cnet_expand_1_1(expand_1_1)
        c_value_list.append(c_expand_1_1)       
        expand_1_2 = self.expand_1_2_source(expand_1_1)
        c_expand_1_2 = self.cnet_expand_1_2(expand_1_2)
        c_value_list.append(c_expand_1_2) 
        deconv_2 = self.deconv_2_source(expand_1_2)
        c_deconv_2 = self.cnet_deconv_2(deconv_2)
        c_value_list.append(c_deconv_2) 

        concat2 = torch.cat([deconv_2, contr_3_2], 1)
        expand_2_1 = self.expand_2_1_source(concat2)
        c_expand_2_1 = self.cnet_expand_2_1(expand_2_1)
        c_value_list.append(c_expand_2_1) 
        expand_2_2 = self.expand_2_2_source(expand_2_1)
        c_expand_2_2 = self.cnet_expand_2_2(expand_2_2)
        c_value_list.append(c_expand_2_2) 
        deconv_3 = self.deconv_3_source(expand_2_2)
        c_deconv_3 = self.cnet_deconv_3(deconv_3)
        c_value_list.append(c_deconv_3) 

        concat3 = torch.cat([deconv_3, contr_2_2], 1)
        expand_3_1 = self.expand_3_1_source(concat3)
        c_expand_3_1 = self.cnet_expand_3_1(expand_3_1)
        c_value_list.append(c_expand_3_1) 
        expand_3_2 = self.expand_3_2_source(expand_3_1)
        c_expand_3_2 = self.cnet_expand_3_2(expand_3_2)
        c_value_list.append(c_expand_3_2) 
        deconv_4 = self.deconv_4_source(expand_3_2)
        c_deconv_4 = self.cnet_deconv_4(deconv_4)
        c_value_list.append(c_deconv_4) 

        concat4 = torch.cat([deconv_4, contr_1_2], 1)
        expand_4_1 = self.expand_4_1_source(concat4)
        c_expand_4_1 = self.cnet_expand_4_1(expand_4_1)
        c_value_list.append(c_expand_4_1) 
        expand_4_2 = self.expand_4_2_source(expand_4_1)
        c_expand_4_2 = self.cnet_expand_4_2(expand_4_2)
        c_value_list.append(c_expand_4_2) 
        ##----conv_5 of the source model: warmup pseudo novel tracts prediction---
        conv_5 = self.conv_5_new_source(expand_4_2)
        c_conv_5 = self.cnet_conv_5(conv_5)
        c_value_list.append(c_conv_5) 
        
        WarmupSeg = F.sigmoid(conv_5)
        return WarmupSeg, c_value_list