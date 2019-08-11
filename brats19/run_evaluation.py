#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 22:55:35 2018

@author: reubendo
"""
import numpy as np
import nibabel 
import os
from argparse import ArgumentParser
import pandas as pd
from definitions import *

gt_path = PREPROCESSED_BRATS_PATH + '/{}Label.nii.gz'


def dice_score(gt, pred):
    true_pos = np.float(np.sum(gt * pred ))
    union = np.float(np.sum(gt) + np.sum(pred))
    if union == 0:
        return 1.
    else:
        dice = true_pos * 2.0 / union
        return dice

def total_score(pred_path, gt_path, file_names):
    print(pred_path)
    score = dict()
    score['subject_id'] = []
    score['whole_tumor'] = []
    score['core_tumor'] = []
    score['enhancing_tumor'] = []
    
    for name in file_names:
        ground_truth = gt_path.format(name)
        ground_truth = os.path.expanduser(ground_truth)
        image_gt = nibabel.load(ground_truth).get_data()
        image_gt = image_gt.reshape(image_gt.shape[:3])
        score['subject_id'].append(name)

        try:
            pred = pred_path.format(name)
            pred = os.path.expanduser(pred)
            image_pred = nibabel.load(pred).get_data()
            image_pred = image_pred.reshape(image_pred.shape[:3])

            # compute Dice for the whole tumor
            set_tumor = [1, 2, 3]
            set_tumor_pred = set_tumor
            image_gt_bin = np.where(np.isin(image_gt, set_tumor)==True, 1, 0)
            image_pred_bin = np.where(np.isin(image_pred, set_tumor_pred)==True, 1, 0)
            dice = dice_score(image_gt_bin, image_pred_bin)
            score['whole_tumor'].append(dice)

            # compute Dice for the core tumor
            set_tumor = [1, 3]
            set_tumor_pred = set_tumor
            image_gt_bin = np.where(np.isin(image_gt, set_tumor)==True , 1, 0)
            image_pred_bin = np.where(np.isin(image_pred, set_tumor_pred)==True, 1, 0)
            dice = dice_score(image_gt_bin, image_pred_bin)
            score['core_tumor'].append(dice)

            # compute Dice for the enhancing tumor
            set_tumor = [3]
            set_tumor_pred = set_tumor
            image_gt_bin = np.where(np.isin(image_gt, set_tumor)==True , 1, 0)
            image_pred_bin = np.where(np.isin(image_pred, set_tumor_pred)==True, 1, 0)
            dice = dice_score(image_gt_bin, image_pred_bin)
            score['enhancing_tumor'].append(dice)

        except Exception as e:
            print(e)
    
    for label in ['whole_tumor', 'core_tumor', 'enhancing_tumor']:
        try:
            print('label: {}, Dice mean: {}+/-{}'.format(label,
                                                         round(100*np.mean(score[label]), 1),
                                                         round(100*np.std(score[label]), 1)))
        except Exception:
            print('label: {}, Dice mean: {}'.format(label, 100*np.mean(score[label])))
    print('%d whole tumor scores have been computed' % len(score['whole_tumor']))
    print('%d core tumor scores have beeen computed' % len(score['core_tumor']))
    print('%d enhancing tumor scores have been computed' % len(score['enhancing_tumor']))
    print(score.keys())
    
    return score

def main(output_path, dataset_split_path, typ):
    # print command line arguments
    try:
        df_split = pd.read_csv(dataset_split_path, header=None)
        if typ == 'all':
            file_names = df_split[0].tolist()
        else:
            file_names = df_split[df_split[1].isin([typ])][0].tolist()
        path_k = output_path + '/window_seg_{}__niftynet_out.nii.gz'
        scores = total_score(path_k, gt_path, file_names)
        df2 = pd.DataFrame(scores, columns=['subject_id', 'whole_tumor', 'core_tumor', 'enhancing_tumor'])
        df2.to_csv(os.path.join(output_path, 'scores.csv'))
    except Exception as e:
        print(e)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--output_dir',
                        help='folder containing the prediction to evaluate',
                        default=os.path.join(NIFTYNET_PATH, 'models_JADE', 'BraTS19', 'u_mvae', 'output2'))
    parser.add_argument('--data_split',
                        help='dataset split csv to use',
                        default=os.path.join(os.path.dirname(PREPROCESSED_BRATS_PATH), 'dataset_split_fold0.csv'))
    parser.add_argument('--data_to_infer',
                        help='part of the data split to valuate. Can be training, validation, inference or all.',
                        default='validation')
    try:
        args = parser.parse_args()
        output_dir = args.output_dir
        data_split_path = args.data_split
        type = args.data_to_infer.lower()
        print('Using %s dataset' % type)
        main(output_dir, data_split_path, type)
    except Exception as e:
        print(e)
