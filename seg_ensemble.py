# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import math
import matplotlib.pyplot as plt
import numpy as np
import openpyxl

import os
from os.path import join
import shutil
import nibabel as nib
import numpy as np

from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy,color_fa

def compute_dice_score(predict, gt):
    overlap = 2.0 * np.sum(predict * gt)
    return overlap / (np.sum(predict) + np.sum(gt))

def compute_iou_score(predict, gt):
    overlap = np.sum(predict * gt)
    return overlap / (np.sum(predict) + np.sum(gt)-overlap)

def compute_rvd_score(predict, gt):
    rvd = abs(1 - np.sum(predict)/np.sum(gt))
    return rvd


####----------------------------- HCP_2.5_34--------------------------------------------------------------------------
test_dataset = "HCP"
seg_size = [72,87,72]
tract_num = 4 # 4.6.12

save_dir = '/data/TractSeg_Run/one_shot_ensemble/HCP_2.5mm_34/Oneshot_845458_'+str(tract_num)
seg_size.append(tract_num)
label_name = 'bundle_masks_'+str(tract_num)+'.nii.gz'

##---------------------for 4 tract---------------------------
Pretrain = "One_step_pretrain" # "One_step"/"Two_step"
seg_dir_list=[
'/data/TractSeg_Run/hcp_exp9/my_custom_experiment_x18/best_weights_ep142.npz',#0,
'/data/TractSeg_Run/hcp_exp9/my_custom_experiment_x19/best_weights_ep213.npz',#1
'/data/TractSeg_Run/hcp_exp9/my_custom_experiment_x20/best_weights_ep234.npz',#2
'/data/TractSeg_Run/hcp_exp12/my_custom_experiment_x21/best_weights_ep273.npz',#3
'/data/TractSeg_Run/hcp_exp11/my_custom_experiment/best_weights_ep157.npz',#4
'/data/TractSeg_Run/hcp_exp11/my_custom_experiment_x2/best_weights_ep285.npz']#5
emsemble_list=[[0,1,4,5], [2,3,4,5], [0,1,2,3,4,5]]


if test_dataset == "HCP":
    Test_peak_dir = '/data/HCP_2.5mm_341k/HCP_for_training_COPY'
    test_sub = ["613538"]

    if tract_num == 4:
        bundles = ['CST_left', 'CST_right', 'OR_left', 'OR_right']
    elif tract_num == 6:
        bundles = ['CST_left', 'CST_right', 'OR_left', 'OR_right', 'POPT_left', 'POPT_right']
    elif tract_num == 12:
        bundles = ['CST_left', 'CST_right', 'FPT_left', 'FPT_right', 'ILF_left', 'ILF_right', 'OR_left', 'OR_right',
                   'POPT_left', 'POPT_right', 'UF_left', 'UF_right']



for sub in test_sub:
    for used_list in emsemble_list:#used_list:[0,1,4,5]
        seg_sum = np.zeros(seg_size)
        emsemble_type = ''
        num_aug = len(used_list)
        print('used_list:', used_list, 'num_aug:', num_aug)
        for used_index in used_list:#0
            used_model = seg_dir_list[used_index]
            save_path = used_model.split('/b')[0]
            test_epoch = used_model.split('/')[-1].split('_')[-1].split('.')[0]
            seg_path = join(save_path, "segmentation" + '_' + test_epoch, 'all_bund_seg', sub+'.nii.gz')
            seg_nii = nib.load(seg_path)
            seg_affine = seg_nii.affine
            seg_data = seg_nii.get_fdata()
            seg_sum += seg_data
            emsemble_type += str(used_index)
        print('emsemble_type', emsemble_type)
        ## 0.5 to 0
        seg_half2zero = (seg_sum > (num_aug/2)).astype(int)
        seg_half2zero_nii = nib.Nifti1Image(seg_half2zero, seg_affine)
        path = os.path.join(save_dir, Pretrain, emsemble_type+'_0.5to0')
        if os.path.exists(path) == 0:
            os.makedirs(path)
        seg_half2zero_path = os.path.join(path, sub+'.nii.gz')
        nib.save(seg_half2zero_nii, seg_half2zero_path)
        ## 0.5 to 1
        seg_half2one = (seg_sum > (num_aug/2-1)).astype(int)
        seg_half2one_nii = nib.Nifti1Image(seg_half2one, seg_affine)
        path = os.path.join(save_dir, Pretrain, emsemble_type+'_0.5to1')
        if os.path.exists(path) == 0:
            os.makedirs(path)
        seg_half2one_path = os.path.join(path, sub+'.nii.gz')
        nib.save(seg_half2one_nii, seg_half2one_path)
        ## 0.5 to random
        seg_rand_onezero = np.random.randint(0,2,seg_size)
        seg_half = (seg_sum == (num_aug/2)).astype(int) * seg_rand_onezero
        seg_halfrand = seg_half2zero + seg_half
        seg_halfrand_nii = nib.Nifti1Image(seg_halfrand, seg_affine)
        path = os.path.join(save_dir, Pretrain, emsemble_type+'_0.5toRand')
        if os.path.exists(path) == 0:
            os.makedirs(path)
        seg_halfrand_path = os.path.join(path, sub+'.nii.gz')
        nib.save(seg_halfrand_nii, seg_halfrand_path)



print('computing evaluation metrics')
all_types = os.listdir(os.path.join(save_dir, Pretrain))
for type in all_types:
    Seg_path = os.path.join(save_dir, Pretrain, type)
    Label_path = Test_peak_dir

    Dice_all = np.zeros([len(test_sub), tract_num])
    print(Dice_all.shape)
    IOU_all = np.zeros([len(test_sub), tract_num])
    RVD_all = np.zeros([len(test_sub), tract_num])

    for subject_index in range(len(test_sub)):
        subject = test_sub[subject_index]
        print("Get_test subject {}".format(subject))
        seg_path = join(Seg_path, subject + ".nii.gz")
        label_path = join(Label_path, subject, label_name)
        print(seg_path)
        print(label_path)
        seg = nib.load(seg_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        for tract_index in range(label.shape[-1]):
            Dice_all[subject_index, tract_index] = compute_dice_score(seg[:, :, :, tract_index],
                                                                      label[:, :, :, tract_index])
            IOU_all[subject_index, tract_index] = compute_iou_score(seg[:, :, :, tract_index], label[:, :, :, tract_index])
            RVD_all[subject_index, tract_index] = compute_rvd_score(seg[:, :, :, tract_index], label[:, :, :, tract_index])

        with open(join(Seg_path, "test_dice.txt"), 'a') as f:
            f.write('Dice of subject {} is \n {} \n'.format(subject, Dice_all[subject_index, :]))
        with open(join(Seg_path, "test_iou.txt"), 'a') as f:
            f.write('IOU of subject {} is \n {} \n'.format(subject, IOU_all[subject_index, :]))
        with open(join(Seg_path, "test_rvd.txt"), 'a') as f:
            f.write('RVD of subject {} is \n {} \n'.format(subject, RVD_all[subject_index, :]))

    Dice_mean = np.mean(Dice_all, 0)
    Dice_average = np.mean(Dice_all)
    IOU_mean = np.mean(IOU_all, 0)
    IOU_average = np.mean(IOU_all)
    RVD_mean = np.mean(RVD_all, 0)
    RVD_average = np.mean(RVD_all)
    with open(join(Seg_path, "test_dice.txt"), 'a') as f:
        for index in range(tract_num):
            log = '{}: {} \n'.format(bundles[index], Dice_mean[index])
            f.write(log)
        log = 'mean dice of all tract is:{}\n'.format(Dice_average)
        f.write(log)
        print(log)

    with open(join(Seg_path, "test_iou.txt"), 'a') as f:
        for index in range(tract_num):
            log = '{}: {} \n'.format(bundles[index], IOU_mean[index])
            f.write(log)
        log = 'mean iou of all tract is:{}\n'.format(IOU_average)
        f.write(log)
        print(log)

    with open(join(Seg_path, "test_rvd.txt"), 'a') as f:
        for index in range(tract_num):
            log = '{}: {} \n'.format(bundles[index], RVD_mean[index])
            f.write(log)
        log = 'mean rvd of all tract is:{}\n'.format(RVD_average)
        f.write(log)
        print(log)

