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



bundles = ['AF_left', 'AF_right', 'ATR_left', 'ATR_right', 'CA', 'CC_1', 'CC_2', 'CC_3', 'CC_4', 'CC_5', 'CC_6',
                   'CC_7', 'CG_left', 'CG_right', 'CST_left', 'CST_right', 'MLF_left', 'MLF_right', 'FPT_left',
                   'FPT_right', 'FX_left', 'FX_right', 'ICP_left', 'ICP_right', 'IFO_left', 'IFO_right', 'ILF_left',
                   'ILF_right', 'MCP', 'OR_left', 'OR_right', 'POPT_left', 'POPT_right', 'SCP_left', 'SCP_right',
                   'SLF_I_left', 'SLF_I_right', 'SLF_II_left', 'SLF_II_right', 'SLF_III_left', 'SLF_III_right',
                   'STR_left', 'STR_right', 'UF_left', 'UF_right', 'CC', 'T_PREF_left', 'T_PREF_right', 'T_PREM_left',
                   'T_PREM_right', 'T_PREC_left', 'T_PREC_right', 'T_POSTC_left', 'T_POSTC_right', 'T_PAR_left',
                   'T_PAR_right', 'T_OCC_left', 'T_OCC_right', 'ST_FO_left', 'ST_FO_right', 'ST_PREF_left',
                   'ST_PREF_right', 'ST_PREM_left', 'ST_PREM_right', 'ST_PREC_left', 'ST_PREC_right', 'ST_POSTC_left',
                   'ST_POSTC_right', 'ST_PAR_left', 'ST_PAR_right', 'ST_OCC_left', 'ST_OCC_right']
#100 HCP
HCP_subjects =["992774", "991267", "987983", "984472", "983773", "979984", "978578", "965771", "965367", "959574",
               "958976", "957974", "951457", "932554", "930449", "922854", "917255", "912447", "910241", "904044",
               "901442", "901139", "901038", "899885", "898176", "896879", "896778", "894673", "889579", "877269",
               "877168", "872764", "872158", "871964", "871762", "865363", "861456", "859671", "857263", "856766",
               "849971", "845458", "837964", "837560", "833249", "833148", "826454", "826353", "816653", "814649",
               "802844", "792766", "792564", "789373", "786569", "784565", "782561", "779370", "771354", "770352",
               "765056", "761957", "759869", "756055", "753251", "751348", "749361", "748662", "748258", "742549",
               "734045", "732243", "729557", "729254", "715647", "715041", "709551", "705341", "704238", "702133",
               "695768", "690152", "687163", "685058", "683256", "680957", "679568", "677968", "673455", "672756",
               "665254", "654754", "645551", "644044", "638049", "627549", "623844", "622236", "620434", "613538"]



## extract 12 tracts
bundles_novel_12 = ['CST_left', 'CST_right', 'FPT_left', 'FPT_right', 'ILF_left', 'ILF_right','OR_left', 'OR_right', 'POPT_left', 'POPT_right',  'UF_left', 'UF_right']
bundles_index_12 = [14, 15, 18, 19, 26, 27, 29, 30, 31, 32, 43, 44]

## extract 6 tracts
bundles_novel_6 = ['CST_left', 'CST_right', 'OR_left', 'OR_right', 'POPT_left', 'POPT_right']
bundles_index_6 = [14, 15, 29, 30, 31, 32]

## extract 4 tracts
bundles_novel_4 = ['CST_left', 'CST_right','OR_left', 'OR_right']
bundles_index_4 = [14, 15, 29, 30]



def extract_bundle_hcp():
    dir = '/data/HCP_2.5mm_341k'
    for sub in HCP_subjects:
        print(sub)
        all_bund_file = os.path.join(dir, sub, 'bundle_masks_72.nii.gz')
        bund_nii =nib.load(all_bund_file)
        affine=bund_nii.affine
        header=bund_nii.header
        bund_data = bund_nii.get_fdata()

        novel_index_12 = np.array(bundles_index_12).astype(int)
        bund_12 = bund_data[:,:,:,novel_index_12].astype(np.float32)
        bundle_nii_12 = nib.Nifti1Image(bund_12, affine, header)
        nib.save(bundle_nii_12, os.path.join(dir, sub, 'bundle_masks_12.nii.gz'))

        novel_index_6 = np.array(bundles_index_6).astype(int)
        bund_6 = bund_data[:,:,:,novel_index_6].astype(np.float32)
        bundle_nii_6 = nib.Nifti1Image(bund_6, affine, header)
        nib.save(bundle_nii_6, os.path.join(dir, sub, 'bundle_masks_6.nii.gz'))

        novel_index_4 = np.array(bundles_index_4).astype(int)
        bund_4 = bund_data[:,:,:,novel_index_4].astype(np.float32)
        bundle_nii_4 = nib.Nifti1Image(bund_4, affine, header)
        nib.save(bundle_nii_4, os.path.join(dir, sub, 'bundle_masks_4.nii.gz'))




if __name__ == "__main__":
    extract_bundle_hcp()
