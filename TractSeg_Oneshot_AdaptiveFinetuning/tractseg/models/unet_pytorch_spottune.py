
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F

from tractseg.libs.pytorch_utils import conv2d
from tractseg.libs.pytorch_utils import deconv2d


class UNet_Pytorch_Spottune(torch.nn.Module):
    def __init__(self, fix_other_layer, img_size, n_input_channels=3, n_classes=7, n_filt=64, batchnorm=False, dropout=False, upsample="bilinear"):
        super(UNet_Pytorch_Spottune, self).__init__()

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

        for p in self.parameters():
            p.requires_grad = False


        ##------------------Target_Trainable----------------------
        self.contr_1_1 = conv2d(n_input_channels, n_filt, batchnorm=batchnorm)
        self.contr_1_2 = conv2d(n_filt, n_filt, batchnorm=batchnorm)
        self.pool_1 = nn.MaxPool2d((2, 2))

        self.contr_2_1 = conv2d(n_filt, n_filt * 2, batchnorm=batchnorm)
        self.contr_2_2 = conv2d(n_filt * 2, n_filt * 2, batchnorm=batchnorm)
        self.pool_2 = nn.MaxPool2d((2, 2))

        self.contr_3_1 = conv2d(n_filt * 2, n_filt * 4, batchnorm=batchnorm)
        self.contr_3_2 = conv2d(n_filt * 4, n_filt * 4, batchnorm=batchnorm)
        self.pool_3 = nn.MaxPool2d((2, 2))

        self.contr_4_1 = conv2d(n_filt * 4, n_filt * 8, batchnorm=batchnorm)
        self.contr_4_2 = conv2d(n_filt * 8, n_filt * 8, batchnorm=batchnorm)
        self.pool_4 = nn.MaxPool2d((2, 2))

        self.dropout = nn.Dropout(p=0.4)

        self.encode_1 = conv2d(n_filt * 8, n_filt * 16, batchnorm=batchnorm)
        self.encode_2 = conv2d(n_filt * 16, n_filt * 16, batchnorm=batchnorm)
        self.deconv_1 = deconv2d(n_filt * 16, n_filt * 16, kernel_size=2, stride=2)

        self.expand_1_1 = conv2d(n_filt * 8 + n_filt * 16, n_filt * 8, batchnorm=batchnorm)
        self.expand_1_2 = conv2d(n_filt * 8, n_filt * 8, batchnorm=batchnorm)
        self.deconv_2 = deconv2d(n_filt * 8, n_filt * 8, kernel_size=2, stride=2)

        self.expand_2_1 = conv2d(n_filt * 4 + n_filt * 8, n_filt * 4, stride=1, batchnorm=batchnorm)
        self.expand_2_2 = conv2d(n_filt * 4, n_filt * 4, stride=1, batchnorm=batchnorm)
        self.deconv_3 = deconv2d(n_filt * 4, n_filt * 4, kernel_size=2, stride=2)

        self.expand_3_1 = conv2d(n_filt * 2 + n_filt * 4, n_filt * 2, stride=1, batchnorm=batchnorm)
        self.expand_3_2 = conv2d(n_filt * 2, n_filt * 2, stride=1, batchnorm=batchnorm)
        self.deconv_4 = deconv2d(n_filt * 2, n_filt * 2, kernel_size=2, stride=2)

        self.expand_4_1 = conv2d(n_filt + n_filt * 2, n_filt, stride=1, batchnorm=batchnorm)
        self.expand_4_2 = conv2d(n_filt, n_filt, stride=1, batchnorm=batchnorm)

        self.conv_5_new = nn.Conv2d(n_filt, n_classes, kernel_size=1, stride=1, padding=0, bias=True)


    # def cvalue_softmax(self, logits):
    #     y = F.softmax(logits, dim=-1)##[b,c,2]
    #     shape = y.size()
    #     _, ind = y.max(dim=-1)
    #     y_hard = torch.zeros_like(y).view(-1, shape[-1])
    #     y_hard.scatter_(1, ind.view(-1, 1), 1)
    #     y_hard = y_hard.view(*shape)
    #     return (y_hard - y).detach() + y


    def forward_slayer_update(self, in_fea, tar_layer, source_layer, action_mask, c_value_list, layer_index):#
        c_value = c_value_list[layer_index-1]##[b,c*2,1,1]--logits

        #*************
        # c_value = self.cvalue_softmax(c_value.view(c_value.shape[0],-1,2))##[b,c,2]
        # c_value = c_value[:,:,1]#[b,c]


        c_action= c_value.contiguous()#[b,c]
        c_action_mask = c_action.view(c_value.shape[0],-1,1,1)#[batch,c,1,1]

        policy_current = action_mask[..., layer_index-1]#mask:[batch 1 1 1];in_fea:[batch,in_c,in_w,in_h]
        feature_sl = source_layer(in_fea) * (1-c_action_mask*policy_current) + tar_layer(in_fea) * (c_action_mask*policy_current)
        return feature_sl


    def forward_slayer(self, in_fea, tar_layer, source_layer, action_mask, layer_index):#
        policy_current = action_mask[..., layer_index-1]#mask:[batch 1 1 1];in_fea:[batch,in_c,in_w,in_h]
        feature_sl = source_layer(in_fea) * (1-policy_current) + tar_layer(in_fea) * policy_current
        return feature_sl


    def forward(self, input, policy, c_value_list):
        action = policy.contiguous()#[batch, layer_n]
        action_mask = action.view(-1, 1, 1, 1, action.shape[1])#[batch,1,1,1,layer_n]

        ### ------------------P and C value------------------
        contr_1_1 = self.forward_slayer_update(input, self.contr_1_1, self.contr_1_1_source, action_mask, c_value_list, 1)
        contr_1_2 = self.forward_slayer_update(contr_1_1, self.contr_1_2, self.contr_1_2_source, action_mask, c_value_list, 2)
        pool_1 = self.pool_1(contr_1_2)

        contr_2_1 = self.forward_slayer_update(pool_1, self.contr_2_1, self.contr_2_1_source, action_mask,c_value_list,  3)
        contr_2_2 = self.forward_slayer_update(contr_2_1, self.contr_2_2, self.contr_2_2_source, action_mask,c_value_list,  4)
        pool_2 = self.pool_2(contr_2_2)

        contr_3_1 = self.forward_slayer_update(pool_2, self.contr_3_1, self.contr_3_1_source, action_mask, c_value_list, 5)
        contr_3_2 = self.forward_slayer_update(contr_3_1, self.contr_3_2, self.contr_3_2_source, action_mask, c_value_list, 6)
        pool_3 = self.pool_3(contr_3_2)

        contr_4_1 = self.forward_slayer_update(pool_3, self.contr_4_1, self.contr_4_1_source, action_mask, c_value_list,7)
        contr_4_2 = self.forward_slayer_update(contr_4_1, self.contr_4_2, self.contr_4_2_source, action_mask, c_value_list, 8)
        pool_4 = self.pool_4(contr_4_2)

        if self.use_dropout:
            pool_4 = self.dropout(pool_4)

        encode_1 = self.forward_slayer_update(pool_4, self.encode_1, self.encode_1_source, action_mask, c_value_list,9)
        encode_2 = self.forward_slayer_update(encode_1, self.encode_2, self.encode_2_source, action_mask, c_value_list, 10)

        ###*********the decoder start***************

        # deconv_1 = self.forward_slayer_update(encode_2, self.deconv_1, self.deconv_1_source, action_mask, c_value_list,11)

        # concat1 = torch.cat([deconv_1, contr_4_2], 1)
        # expand_1_1 = self.forward_slayer_update(concat1, self.expand_1_1, self.expand_1_1_source, action_mask,c_value_list, 12) 
        # expand_1_2 = self.forward_slayer_update(expand_1_1, self.expand_1_2, self.expand_1_2_source, action_mask, c_value_list,13)
        # deconv_2 = self.forward_slayer_update(expand_1_2, self.deconv_2, self.deconv_2_source, action_mask, c_value_list,14)

        # concat2 = torch.cat([deconv_2, contr_3_2], 1)
        # expand_2_1 = self.forward_slayer_update(concat2, self.expand_2_1, self.expand_2_1_source, action_mask, c_value_list,15) 
        # expand_2_2 = self.forward_slayer_update(expand_2_1, self.expand_2_2, self.expand_2_2_source, action_mask,c_value_list, 16)
        # deconv_3 = self.forward_slayer_update(expand_2_2, self.deconv_3, self.deconv_3_source, action_mask, c_value_list,17)

        # concat3 = torch.cat([deconv_3, contr_2_2], 1)
        # expand_3_1 = self.forward_slayer_update(concat3, self.expand_3_1, self.expand_3_1_source, action_mask,c_value_list, 18) 
        # expand_3_2 = self.forward_slayer_update(expand_3_1, self.expand_3_2, self.expand_3_2_source, action_mask, c_value_list,19)
        # deconv_4 = self.forward_slayer_update(expand_3_2, self.deconv_4, self.deconv_4_source, action_mask, c_value_list,20)

        # concat4 = torch.cat([deconv_4, contr_1_2], 1)
        # expand_4_1 = self.forward_slayer_update(concat4, self.expand_4_1, self.expand_4_1_source, action_mask,c_value_list, 21) 
        # expand_4_2 = self.forward_slayer_update(expand_4_1, self.expand_4_2, self.expand_4_2_source, action_mask,c_value_list, 22)
        # conv_5 = self.forward_slayer_update(expand_4_2, self.conv_5_new, self.conv_5_new_source, action_mask,c_value_list, 23)



        # #### ------------------only p value--------------------------------
        # # contr_1_1 = self.forward_slayer(input, self.contr_1_1, self.contr_1_1_source, action_mask, 1)
        # # contr_1_2 = self.forward_slayer(contr_1_1, self.contr_1_2, self.contr_1_2_source, action_mask, 2)
        # # pool_1 = self.pool_1(contr_1_2)

        # # contr_2_1 = self.forward_slayer(pool_1, self.contr_2_1, self.contr_2_1_source, action_mask,  3)
        # # contr_2_2 = self.forward_slayer(contr_2_1, self.contr_2_2, self.contr_2_2_source, action_mask,  4)
        # # pool_2 = self.pool_2(contr_2_2)

        # # contr_3_1 = self.forward_slayer(pool_2, self.contr_3_1, self.contr_3_1_source, action_mask, 5)
        # # contr_3_2 = self.forward_slayer(contr_3_1, self.contr_3_2, self.contr_3_2_source, action_mask, 6)
        # # pool_3 = self.pool_3(contr_3_2)

        # # contr_4_1 = self.forward_slayer(pool_3, self.contr_4_1, self.contr_4_1_source, action_mask,7)
        # # contr_4_2 = self.forward_slayer(contr_4_1, self.contr_4_2, self.contr_4_2_source, action_mask, 8)
        # # pool_4 = self.pool_4(contr_4_2)

        # # if self.use_dropout:
        # #     pool_4 = self.dropout(pool_4)

        # encode_1 = self.forward_slayer(pool_4, self.encode_1, self.encode_1_source, action_mask,9)
        # encode_2 = self.forward_slayer(encode_1, self.encode_2, self.encode_2_source, action_mask, 10)

        ###*********the decoder start***************

        deconv_1 = self.forward_slayer(encode_2, self.deconv_1, self.deconv_1_source, action_mask,11)

        concat1 = torch.cat([deconv_1, contr_4_2], 1)
        expand_1_1 = self.forward_slayer(concat1, self.expand_1_1, self.expand_1_1_source, action_mask, 12) 
        expand_1_2 = self.forward_slayer(expand_1_1, self.expand_1_2, self.expand_1_2_source, action_mask,13)
        deconv_2 = self.forward_slayer(expand_1_2, self.deconv_2, self.deconv_2_source, action_mask,14)

        concat2 = torch.cat([deconv_2, contr_3_2], 1)
        expand_2_1 = self.forward_slayer(concat2, self.expand_2_1, self.expand_2_1_source, action_mask,15) 
        expand_2_2 = self.forward_slayer(expand_2_1, self.expand_2_2, self.expand_2_2_source, action_mask, 16)
        deconv_3 = self.forward_slayer(expand_2_2, self.deconv_3, self.deconv_3_source, action_mask,17)

        concat3 = torch.cat([deconv_3, contr_2_2], 1)
        expand_3_1 = self.forward_slayer(concat3, self.expand_3_1, self.expand_3_1_source, action_mask, 18) 
        expand_3_2 = self.forward_slayer(expand_3_1, self.expand_3_2, self.expand_3_2_source, action_mask,19)
        deconv_4 = self.forward_slayer(expand_3_2, self.deconv_4, self.deconv_4_source, action_mask,20)

        concat4 = torch.cat([deconv_4, contr_1_2], 1)
        expand_4_1 = self.forward_slayer(concat4, self.expand_4_1, self.expand_4_1_source, action_mask, 21) 
        expand_4_2 = self.forward_slayer(expand_4_1, self.expand_4_2, self.expand_4_2_source, action_mask, 22)
        conv_5 = self.forward_slayer(expand_4_2, self.conv_5_new, self.conv_5_new_source, action_mask, 23)


        
        return conv_5




    # def flops_conv3_relu(self, in_size, in_channel, out_channel, kernel_size=3, stride=1):
    #     out_size = in_size
    #     ## conv3*3 + bias
    #     flops = (out_size**2)*((kernel_size**2)*in_channel*out_channel + out_channel)
    #     ## relu
    #     flops += (out_size**2)*out_channel
    #     return flops
    # def flops_conv1(self,in_size, in_channel, out_channel, kernel_size=1, stride=1):
    #     out_size = in_size
    #     ## conv1*1 + bias
    #     flops = (out_size**2)*((kernel_size**2)*in_channel*out_channel + out_channel)
    #     return flops
    # def flops_deconv2_relu(self, in_size, in_channel, out_channel, kernel_size=2, stride=2):
    #     out_size = in_size*kernel_size
    #     ## deconv2*2 + bias
    #     flops = (out_size**2)*((kernel_size**2)*in_channel*out_channel + out_channel)
    #     ## relu
    #     flops += (out_size**2)*out_channel
    #     return flops
    # def flops_maxpool(self, in_size, in_channel, kernel_size=2, stride=2):
    #     ## number of elements
    #     flops = (in_size**2)*in_channel
    #     return flops

    # def flops(self, downsample=4):
    #     flops = 0
    #     ## encoder (4 block: 2conv+maxpool)
    #     for block_index in range(downsample):
    #         if block_index == 0:
    #             flops += self.flops_conv3_relu(in_size=self.img_size / (2 ** block_index),
    #                                            in_channel=self.in_channel,
    #                                            out_channel=self.n_filt * (2 ** block_index))
    #         else:
    #             flops += self.flops_conv3_relu(in_size=self.img_size / (2 ** block_index),
    #                                            in_channel=self.n_filt * (2 ** (block_index - 1)),
    #                                            out_channel=self.n_filt * (2 ** block_index))
    #         flops += self.flops_conv3_relu(in_size=self.img_size / (2 ** block_index),
    #                                        in_channel=self.n_filt * (2 ** block_index),
    #                                        out_channel=self.n_filt * (2 ** block_index))
    #         flops += self.flops_maxpool(in_size=self.img_size / (2 ** block_index),
    #                               in_channel=self.n_filt * (2 ** block_index))
    #     ## bottleneck
    #     flops += self.flops_conv3_relu(in_size=self.img_size / (2 ** downsample),
    #                                    in_channel=self.n_filt * (2 ** (downsample-1)),
    #                                    out_channel=self.n_filt * (2 ** downsample))
    #     flops += self.flops_conv3_relu(in_size=self.img_size / (2 ** downsample),
    #                                    in_channel=self.n_filt * (2 ** downsample),
    #                                    out_channel=self.n_filt * (2 ** downsample))
    #     flops += self.flops_deconv2_relu(in_size=self.img_size / (2 ** downsample),
    #                                    in_channel=self.n_filt * (2 ** downsample),
    #                                    out_channel=self.n_filt * (2 ** downsample))
    #     ## decoder
    #     for block in range(downsample):
    #         block_index = downsample-1-block
    #         flops += self.flops_conv3_relu(in_size=self.img_size / (2 ** block_index),
    #                                        in_channel=self.n_filt * (2**block_index + 2**(block_index+1)),
    #                                        out_channel=self.n_filt * (2 ** block_index))
    #         flops += self.flops_conv3_relu(in_size=self.img_size / (2 ** block_index),
    #                                        in_channel=self.n_filt * (2 ** block_index),
    #                                        out_channel=self.n_filt * (2 ** block_index))
    #         if block_index>0:
    #             flops += self.flops_deconv2_relu(in_size=self.img_size / (2 ** block_index),
    #                                              in_channel=self.n_filt * (2 ** block_index),
    #                                              out_channel=self.n_filt * (2 ** block_index))
    #         else:
    #             flops += self.flops_conv1(in_size = self.img_size, in_channel = self.n_filt, out_channel = self.n_classes)
    #     return flops
