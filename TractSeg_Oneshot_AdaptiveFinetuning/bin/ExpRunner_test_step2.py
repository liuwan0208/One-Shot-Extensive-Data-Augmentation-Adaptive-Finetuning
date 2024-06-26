#!/usr/bin/env python

"""
This module is for training the model. See Readme.md for more details about training your own model.

Examples:
    Run local:
    $ ExpRunner --config=XXX

    Predicting with new config setup:
    $ ExpRunner --train=False --test=True --lw --config=XXX
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import os
import importlib
import argparse
import pickle as pkl
from pprint import pprint
import distutils.util
from os.path import join
import os

import nibabel as nib
import numpy as np
import time
import torch

from tractseg.libs import data_utils
from tractseg.libs import direction_merger
from tractseg.libs import exp_utils
from tractseg.libs import img_utils
from tractseg.libs import peak_utils
from tractseg.libs.system_config import SystemConfig as C
from tractseg.libs import trainer
from tractseg.data.data_loader_training import DataLoaderTraining as DataLoaderTraining2D
from tractseg.data.data_loader_training_3D import DataLoaderTraining as DataLoaderTraining3D
from tractseg.data.data_loader_inference import DataLoaderInference
from tractseg.data import dataset_specific_utils
from tractseg.models.base_model import BaseModel

# from bin.utils_add import compute_dice_score

warnings.simplefilter("ignore", UserWarning)  # hide scipy warnings
warnings.simplefilter("ignore", FutureWarning)  # hide h5py warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")  # hide Cython benign warning
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")  # hide Cython benign warning


def compute_dice_score(predict, gt):
    overlap = 2.0 * np.sum(predict * gt)
    return overlap / (np.sum(predict) + np.sum(gt))


def compute_iou_score(predict, gt):
    overlap = np.sum(predict * gt)
    return overlap / (np.sum(predict) + np.sum(gt) - overlap)


def compute_rvd_score(predict, gt):
    rvd = abs(1 - np.sum(predict) / np.sum(gt))
    return rvd


def main():
    parser = argparse.ArgumentParser(description="Train a network on your own data to segment white matter bundles.",
                                     epilog="Written by Jakob Wasserthal. Please reference 'Wasserthal et al. "
                                            "TractSeg - Fast and accurate white matter tract segmentation. "
                                            "https://doi.org/10.1016/j.neuroimage.2018.07.070)'")
    parser.add_argument("--config", metavar="name", help="Name of configuration to use",
                        default='my_custom_experiment')
    parser.add_argument("--train", metavar="True/False", help="Train network",
                        type=distutils.util.strtobool, default=False)
    parser.add_argument("--test", metavar="True/False", help="Test network",
                        type=distutils.util.strtobool, default=True)
    parser.add_argument("--seg", action="store_true", help="Create binary segmentation", default=True)
    parser.add_argument("--probs", action="store_true", help="Create probmap segmentation")
    parser.add_argument("--lw", action="store_true", help="Load weights of pretrained net")
    parser.add_argument("--only_val", action="store_true", help="only run validation")
    parser.add_argument("--en", metavar="name", help="Experiment name")
    parser.add_argument("--fold", metavar="N", help="Which fold to train when doing CrossValidation", type=int)
    parser.add_argument("--verbose", action="store_true", help="Show more intermediate output", default=True)
    args = parser.parse_args()

    Config = getattr(importlib.import_module("tractseg.experiments.base_step2"), "Config")()
    if args.config:
        # Config.__dict__ does not work properly therefore use this approach
        Config = getattr(importlib.import_module("tractseg.experiments.custom." + args.config), "Config")()

    if args.en:
        Config.EXP_NAME = args.en  ## my_custom_experiment

    Config.TRAIN = bool(args.train)  ## True
    Config.TEST = bool(args.test)  ## False
    Config.SEGMENT = args.seg  ## False
    if args.probs:
        Config.GET_PROBS = True  ## False
    if args.lw:
        Config.LOAD_WEIGHTS = args.lw  ## False
    if args.fold:
        Config.CV_FOLD = args.fold  ## 0
    if args.only_val:
        Config.ONLY_VAL = True  ## False
    Config.VERBOSE = args.verbose  ## True:show more intermedia output

    Config.MULTI_PARENT_PATH = join(C.EXP_PATH, Config.EXP_MULTI_NAME)  ## '//home/wanliu/TractSeg/hcp_exp/'
    Config.EXP_PATH = join(C.EXP_PATH, Config.EXP_MULTI_NAME,
                           Config.EXP_NAME)  ## '/home/wanliu/TractSeg/hcp_exp/my_custom_experiment'

    ##### modify subject numbers
    Config.TRAIN_SUBJECTS, Config.VALIDATE_SUBJECTS, Config.TEST_SUBJECTS = dataset_specific_utils.get_cv_fold(
        Config.CV_FOLD,
        dataset=Config.DATASET)
    Config.TRAIN_SUBJECTS = Config.TRAIN_SUBJECTS[0:20]

    if Config.WEIGHTS_PATH == "":
        Config.WEIGHTS_PATH = exp_utils.get_best_weights_path(Config.EXP_PATH, Config.LOAD_WEIGHTS)

    # Autoset input dimensions based on settings
    Config.INPUT_DIM = dataset_specific_utils.get_correct_input_dim(Config)  ## (144,144)
    Config = dataset_specific_utils.get_labels_filename(Config)  ## 'LABELS_FILENAME': 'bundle_masks_72'

    if Config.EXPERIMENT_TYPE == "peak_regression":
        Config.NR_OF_CLASSES = 3 * len(dataset_specific_utils.get_bundle_names(Config.CLASSES)[1:])
    else:
        Config.NR_OF_CLASSES = len(dataset_specific_utils.get_bundle_names(Config.CLASSES)[1:])  ## 72

    if Config.TRAIN and not Config.ONLY_VAL:
        Config.EXP_PATH = exp_utils.create_experiment_folder(Config.EXP_NAME, Config.MULTI_PARENT_PATH, Config.TRAIN)

    if Config.TRAIN:
        if Config.VERBOSE:
            print("Hyperparameters:")
            exp_utils.print_Configs(Config)

        with open(join(Config.EXP_PATH, "Hyperparameters.txt"), "a") as f:
            Config_dict = {attr: getattr(Config, attr) for attr in dir(Config)
                           if not callable(getattr(Config, attr)) and not attr.startswith("__")}
            pprint(Config_dict, f)
        if Config.DIM == "2D":
            data_loader = DataLoaderTraining2D(Config)
        else:
            data_loader = DataLoaderTraining3D(Config)
        print("Training...")
        model = BaseModel(Config)
        trainer.train_model(Config, model, data_loader)
    Config = exp_utils.get_correct_labels_type(Config)

    test_sub = ["802844", "792766", "792564", "789373", "786569", "784565", "782561", "779370", "771354", "770352",
               "695768", "690152", "687163", "685058", "683256", "680957", "679568", "677968", "673455", "672756",
               "665254", "654754", "645551", "644044", "638049", "627549", "623844", "622236", "620434", "613538"]
    Config.TEST_SUBJECTS = test_sub
    tract_num = Config.NR_OF_CLASSES
    if Config.TEST:

        save_path = '/data7/wanliu/TractSeg_Run/hcp_exp6/my_custom_experiment_x22'
        Test_peak_dir = '/data4/wanliu/HCP_2.5mm_341k/HCP_for_training_COPY'
        for file in os.listdir(save_path):
            # print(file)
            if 'best_weights' in file:
                print(file)
                seg_name = file
            if 'best_CNETweights' in file:
                print(file)
                cnet_name = file
            if 'best_PNETweights' in file:
                print(file)
                pnet_name = file
            
        Config.WEIGHTS_PATH ={'seg':save_path+'/'+seg_name,'cnet':save_path+'/'+cnet_name,'pnet':save_path+'/'+pnet_name}


        inference = True
        cal_metric = True

        seg_path = Config.WEIGHTS_PATH['seg']
        test_epoch = seg_path.split('/')[-1].split('_')[-1].split('.')[0]
        model = BaseModel(Config, inference=True)
        # model.load_model(Config.WEIGHTS_PATH)
        model.load_model_update(Config.WEIGHTS_PATH)

        all_subjects = Config.TEST_SUBJECTS
        if inference:
            inf_time_all = np.zeros([len(all_subjects)])
            for i, subject in enumerate(all_subjects):
                print("Get_segmentation subject {}".format(subject))
                peak_path = join(Test_peak_dir, subject, 'mrtrix_peaks.nii.gz')

                start_time = time.time()
                data_img = nib.load(peak_path)
                data_affine = data_img.affine
                data0 = data_img.get_fdata()
                data = np.nan_to_num(data0)
                data, _, bbox, original_shape = data_utils.crop_to_nonzero(data)
                data, transformation = data_utils.pad_and_scale_img_to_square_img(data, target_size=Config.INPUT_DIM[0],
                                                                                  nr_cpus=-1)

                seg_xyz, _ = direction_merger.get_seg_single_img_3_directions(Config, model, data=data,
                                                                                  scale_to_world_shape=False,
                                                                                  only_prediction=True,
                                                                                  batch_size=1)
                seg = direction_merger.mean_fusion(Config.THRESHOLD, seg_xyz, probs=False)
                seg = data_utils.cut_and_scale_img_back_to_original_img(seg, transformation, nr_cpus=-1)
                seg = data_utils.add_original_zero_padding_again(seg, bbox, original_shape, Config.NR_OF_CLASSES)
                inf_time = time.time() - start_time
                inf_time_all[i] = inf_time

                print('save segmentation results')
                img_seg = nib.Nifti1Image(seg.astype(np.uint8), data_affine)
                output_all_bund = join(save_path, "segmentation" + '_' + test_epoch, 'all_bund_seg')
                exp_utils.make_dir(output_all_bund)
                print(output_all_bund)
                nib.save(img_seg, join(output_all_bund, subject + ".nii.gz"))

                bundles = dataset_specific_utils.get_bundle_names(Config.CLASSES)[1:]
                output_indiv_bund = join(save_path, "segmentation" + '_' + test_epoch, 'indiv_bund_seg', subject)
                exp_utils.make_dir(output_indiv_bund)
                print(output_indiv_bund)
                for idx, bundle in enumerate(bundles):
                    img_seg = nib.Nifti1Image(seg[:, :, :, idx], data_affine)
                    nib.save(img_seg, join(output_indiv_bund, bundle + ".nii.gz"))

            print('mean inference time of a single subject is:', np.mean(inf_time_all),'s')


        ## -----------------------Compute evaluation metrics------------------------
        if cal_metric:
            print('computing evaluation metrics')

            if tract_num==10:
                bundles = ['CST_left', 'CST_right', 'FPT_left', 'FPT_right', 'OR_left', 'OR_right', 'POPT_left', 'POPT_right', 'UF_left', 'UF_right']
            elif tract_num==4:
                bundles = ['CST_left', 'CST_right', 'OR_left', 'OR_right']
            elif tract_num==6:
                bundles = ['CST_left', 'CST_right', 'OR_left', 'OR_right', 'POPT_left', 'POPT_right']


            Seg_path = join(save_path, "segmentation" + '_' + test_epoch, 'all_bund_seg')
            Label_path = Test_peak_dir

            Dice_all = np.zeros([len(all_subjects), tract_num])
            print(Dice_all.shape)
            IOU_all = np.zeros([len(all_subjects), tract_num])
            RVD_all = np.zeros([len(all_subjects), tract_num])

            print(C.DATA_PATH)
            for subject_index in range(len(all_subjects)):
                subject = all_subjects[subject_index]
                print("Get_test subject {}".format(subject))
                seg_path = join(Seg_path, subject + ".nii.gz")
                label_path = join(Label_path, subject, Config.LABELS_FILENAME + '.nii.gz')
                print(seg_path)
                print(label_path)
                seg = nib.load(seg_path).get_fdata()
                label= nib.load(label_path).get_fdata()

                for tract_index in range(label.shape[-1]):
                    Dice_all[subject_index, tract_index] = compute_dice_score(seg[:,:,:,tract_index], label[:,:,:,tract_index])

                with open(join(save_path, "test_dice_"+str(test_epoch)+".txt"), 'a') as f:
                    f.write('Dice of subject {} is \n {} \n'.format(subject, Dice_all[subject_index, :]))


            Dice_mean = np.mean(Dice_all, 0)
            Dice_average = np.mean(Dice_all)

            with open(join(save_path, "test_dice_"+str(test_epoch)+".txt"),'a') as f:
                for index in range(tract_num):
                    log = '{}: {} \n'.format(bundles[index], Dice_mean[index])
                    f.write(log)
                log = 'mean dice of all tract is:{}\n'.format(Dice_average)
                f.write(log)
                print(log)




if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    main()

