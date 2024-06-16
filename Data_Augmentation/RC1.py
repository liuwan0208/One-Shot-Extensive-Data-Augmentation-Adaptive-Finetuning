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



def generate_bbox_cutout(img_size):
	## select weight: lamda--(0,1) uniform distribution
	alpha = 1
	lamda = np.random.beta(alpha, alpha)
	## generate crop 3D patches coordinates: center points--(0, img_szie) uniform dis
	W = img_size[0]
	H = img_size[1]
	D = img_size[2]
	cut_rate = np.sqrt(1 - lamda)
	cut_x = np.int(W * cut_rate)  # box size in each dimension
	cut_y = np.int(H * cut_rate)
	cut_z = np.int(D * cut_rate)
	cx = np.random.randint(W)  # coordinate of box center
	cy = np.random.randint(H)
	cz = np.random.randint(D)
	bbx1 = np.clip(cx - cut_x // 2, 0, W)
	bbx2 = np.clip(cx + cut_x // 2, 0, W)
	bby1 = np.clip(cy - cut_y // 2, 0, H)
	bby2 = np.clip(cy + cut_y // 2, 0, H)
	bbz1 = np.clip(cz - cut_z // 2, 0, D)
	bbz2 = np.clip(cz + cut_z // 2, 0, D)
	return bbx1,bbx2,bby1,bby2,bbz1,bbz2







init_path='/data/HCP_2.5mm_341k/HCP_for_training_COPY'
subjects = ['620434']
labeled_tracts = 4 # 4,6,12
bundle_mask = 'bundle_masks_'+str(labeled_tracts)+'.nii.gz'
log_save_file= '/data/HCP_2.5mm_341k/HCP_daug/Oneshot_620434_4/CutoutYChange_'+str(labeled_tracts)+'tracts.txt'
target_path  = '/data/HCP_2.5mm_341k/HCP_daug/Oneshot_620434_4/CutoutYChange/HCP_for_training_COPY'


used_data_num = len(subjects)
aug_data_num = 15
start_index = 0


sub = subjects[0]
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
	bbx1,bbx2,bby1,bby2,bbz1,bbz2 = generate_bbox_cutout(data.shape)

	log = "Daug_image:{}, sub:{}, cutout region:{}\n".format(i + start_index, sub, [bbx1,bbx2,bby1,bby2,bbz1,bbz2])
	print(log)
	with open(log_save_file, 'a') as f:
		f.write(log)

	data[bbx1:bbx2, bby1:bby2, bbz1:bbz2, :] = 0
	data = data.astype(np.float32)
	data_nii = nib.Nifti1Image(data, data_affine)
	nib.save(data_nii, join(save_dir, 'mrtrix_peaks.nii.gz'))


	label[bbx1:bbx2, bby1:bby2, bbz1:bbz2, :] = 0
	label = label.astype(np.float32)
	label_nii = nib.Nifti1Image(label, label_affine)
	nib.save(label_nii, join(save_dir, bundle_mask))
