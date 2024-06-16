from __future__ import division
import os
import nibabel as nib
import nibabel.processing
import numpy as np
from shutil import copyfile
import shutil
from os.path import join
from random import sample
from dipy.core.gradients import gradient_table
import copy
import math


def int2bin(input_int, length):
	bin_num = bin(input_int) # '0b...'
	ori_len = len(bin_num)-2
	zero_num = length - ori_len # add zero at the beginning of the string
	add_zeros = ''
	for i in range(zero_num):
		add_zeros += '0'
	ori_bin = bin_num[2:] # remove the first two characters
	new_bin = add_zeros + ori_bin
	return new_bin

def bin2list(input_bin):
	len_bin = len(input_bin)
	list = []
	for i in range(len_bin):
		ele = int(input_bin[i])
		list.append(ele)
	return list



### for hcp
init_path='/data/HCP_2.5mm_341k/HCP_for_training_COPY'
subjects = ['620434']
labeled_tracts = 4 # 4,6,12
bundle_mask = 'bundle_masks_'+str(labeled_tracts)+'.nii.gz'
log_save_file= '/data/HCP_2.5mm_341k/HCP_daug/Oneshot_620434_4/MtractoutYChange_'+str(labeled_tracts)+'tracts.txt'
target_path  = '/data/HCP_2.5mm_341k/HCP_daug/Oneshot_620434_4/MtractoutYChange/HCP_for_training_COPY'




used_data_num = len(subjects)
aug_num = 15
data_name = 'mtractout'
start_index = 0



sub = subjects[0]
total_comb = math.pow(2, labeled_tracts)-1 # 2^n-1
aug_data_num = int(min(total_comb, aug_num))
list = range(1, int(total_comb+1)) # 1~(2^n-1)
print(total_comb, list)
sampled_num = sample(list, aug_data_num)
print('aug_data_num:', aug_data_num, '\n', 'total_comb:', total_comb, '\n','sampled_num:', sampled_num, '\n')

for i in range(aug_data_num):
	# load data and label
	data_file = join(init_path, sub, 'mrtrix_peaks.nii.gz')
	data_nii = nib.load(data_file)
	data_affine = data_nii.affine
	data = data_nii.get_fdata()

	label_file = join(init_path, sub, bundle_mask)
	label_nii = nib.load(label_file)
	label_affine = label_nii.affine
	label = label_nii.get_fdata()

	# make_dir
	save_dir = join(target_path, 'daug_' + str(i + start_index))
	if os.path.exists(save_dir) == 0:
		os.makedirs(save_dir)

	## generate mask
	tract_comb = sampled_num[i]
	trial_int = tract_comb
	trial_bin = int2bin(trial_int, length=labeled_tracts)
	trials = bin2list(trial_bin)
	print(trial_bin, trials, np.sum(np.array(trials)))


	log = "Daug_image:{}, sub:{}, tract_comb:{}, trials{}\n".format(i + start_index, sub, tract_comb, trials)
	print(log)
	with open(log_save_file, 'a') as f:
		f.write(log)


	mask_SingleChannel = (np.sum(label * np.array(trials), axis=-1) > 0).astype(int)  # (x,y,z)
	nib.save(nib.Nifti1Image(mask_SingleChannel.astype(float), affine=label_affine), join(save_dir, 'mask_3d.nii.gz'))

	channel_num_data = data.shape[-1]
	mask_data = np.expand_dims(mask_SingleChannel, axis=-1).repeat(channel_num_data, axis=-1)  # [x,y,z,9]
	mask_label = np.expand_dims(mask_SingleChannel, axis=-1).repeat(labeled_tracts, axis=-1)  # (x,y,z,12)

	data = data * (1-mask_data)
	data = data.astype(np.float32)
	label = label * (1-mask_label)
	label = label.astype(np.float32)

	data_nii = nib.Nifti1Image(data, data_affine)
	nib.save(data_nii, join(save_dir, 'mrtrix_peaks.nii.gz'))
	label_nii = nib.Nifti1Image(label, label_affine)
	nib.save(label_nii, join(save_dir, bundle_mask))
