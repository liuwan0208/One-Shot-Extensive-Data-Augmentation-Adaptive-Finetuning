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


subjects = ["992774", "991267", "987983", "984472", "983773", "979984", "978578", "965771", "965367", "959574",
            "958976", "957974", "951457", "932554", "930449", "922854", "917255", "912447", "910241",
            "904044", "901442", "901139", "901038", "899885", "898176", "896879", "896778", "894673", "889579",
            "877269", "877168", "872764", "872158", "871964", "871762", "865363", "861456", "859671",
            "857263", "856766", "849971", "845458", "837964", "837560", "833249", "833148", "826454", "826353",
            "816653", "814649", "802844", "792766", "792564", "789373", "786569", "784565", "782561", "779370",
            "771354", "770352", "765056", "761957", "759869", "756055", "753251", "751348", "749361", "748662",
            "748258", "742549", "734045", "732243", "729557", "729254", "715647", "715041", "709551", "705341",
            "704238", "702133", "695768", "690152", "687163", "685058", "683256", "680957", "679568", "677968",
            "673455", "672756", "665254", "654754", "645551", "644044", "638049", "627549", "623844", "622236",
            "620434", "613538"]##100


##----------- remove the non-brain region of peaks and labels in the HCP_training_COPY file to HCP_preproc----------
def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def remove_nonbrain_area_HCP():
    ori_dir = '/data/HCP_2.5mm_341k/HCP_daug/Oneshot_620434_4/MtractoutYChange/HCP_for_training_COPY'
    new_dir = '/data/HCP_2.5mm_341k/HCP_daug/Oneshot_620434_4/MtractoutYChange/HCP_preproc'
    help_file = 'mrtrix_peaks.nii.gz'
    target_file_list = ['mrtrix_peaks.nii.gz', 'bundle_masks_4.nii.gz']
    subjects=os.listdir(ori_dir)
    for sub in subjects:
        print(sub)
        for target_file in target_file_list:
            ## extract crop area
            help_path = join(ori_dir, sub, help_file)
            help_data = np.nan_to_num(nib.load(help_path).get_fdata())
            bbox = get_bbox_from_mask(help_data)

            ## load the data
            ori_path = join(ori_dir, sub, target_file)
            new_path = join(new_dir, sub, target_file)
            large_nii = nib.load(ori_path)
            large_data = np.nan_to_num(large_nii.get_fdata())
            affine = large_nii.affine

            ## crop non-brain area
            data = large_data[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1],:]
            data_nii = nib.Nifti1Image(data.astype(np.float32), affine = affine)
            if os.path.exists(join(new_dir, sub))==0:
                os.makedirs(join(new_dir, sub))
            nib.save(data_nii, new_path)




if __name__ == "__main__":
    remove_nonbrain_area_HCP()

