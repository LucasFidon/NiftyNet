import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from definitions import *

# START_FROM_ITER = 40000
START_FROM_ITER = 20000
STOP_BEFORE_ITER = 60000
PREDICT_FULL_TRAINING_FOR_BEST_CKPT = True

def find_list_of_checkpoints_available(model_dir):
    check_list = []
    checkpoint_path = os.path.join(model_dir, 'models')
    assert os.path.exists(checkpoint_path), 'Folder %s not found.' % checkpoint_path
    for file in os.listdir(checkpoint_path):
        if '.index' in file:
            checkpoint = int(file.replace('.index', '').replace('model.ckpt-', ''))
            if checkpoint >= START_FROM_ITER and checkpoint <= STOP_BEFORE_ITER:
                check_list.append(checkpoint)
    check_list.sort()
    return check_list

def main(model_dir, data_split_file):
    assert os.path.exists(model_dir), 'Folder %s not found. Please check your --model_dir'
    checkpoint_list = find_list_of_checkpoints_available(model_dir)
    best_cpkt = -1
    best_mean_dice = -1
    best_scores = None

    for ckpt in checkpoint_list:
        # run inference
        output_dir = os.path.join(model_dir, 'output_iter%d' % ckpt)
        # ignore inference if the output folder already exists
        if not os.path.exists(output_dir):
            cmd = '%s/net_run.py inference ' % NIFTYNET_PATH
            cmd += '-a niftynet.extension.variational_segmentation_application.VariationalSegmentationApplication '
            cmd += '-c %s/brats19/config_inf.ini ' % NIFTYNET_PATH
            cmd += '--model_dir %s ' % model_dir
            cmd += '--dataset_split_file %s ' % data_split_file
            cmd += '--dataset_to_infer validation '
            cmd += '--inference_iter %d ' % ckpt
            cmd += '--save_seg_dir ./output_iter%d ' % ckpt
            print('run inference for iteration %d...' % ckpt)
            os.system(cmd)
        else:
            print('%s already exists.' % output_dir)
            print('Skip inference fo iteration %d.' % ckpt)

        # run evaluation
        assert os.path.exists(output_dir), 'Something went wrong with the prediction of iter %d' % ckpt
        cmd = '%s/brats19/run_evaluation.py ' % NIFTYNET_PATH
        cmd += '--output_dir %s ' % output_dir
        cmd += '--data_split %s ' % data_split_file
        cmd += '--data_to_infer validation '
        print('run validation evaluation for iteration %d...' % ckpt)
        os.system(cmd)
        print('')

        # load evaluation csv and compute the mean dice
        scores = pd.read_csv(os.path.join(output_dir, 'scores.csv'))
        mean_dice = 0
        for roi in ['whole_tumor', 'core_tumor', 'enhancing_tumor']:
            mean_dice += np.mean(scores[roi]) / 3.
        if mean_dice > best_mean_dice:
            print('###################################################')
            print('--- New best checkpoint! (iter %d, mead DSC=%f) ---' % (ckpt, mean_dice))
            print('###################################################')
            best_cpkt = ckpt
            best_mean_dice = mean_dice
            best_scores = scores

    # run inference for the training data for the best model
    if PREDICT_FULL_TRAINING_FOR_BEST_CKPT:
        output_dir = os.path.join(model_dir, 'output_best')
        if not os.path.exists(output_dir):
            cmd = '%s/net_run.py inference ' % NIFTYNET_PATH
            cmd += '-a niftynet.extension.variational_segmentation_application.VariationalSegmentationApplication '
            cmd += '-c %s/brats19/config_inf.ini ' % NIFTYNET_PATH
            cmd += '--model_dir %s ' % model_dir
            cmd += '--dataset_split_file %s ' % data_split_file
            cmd += '--dataset_to_infer training '
            cmd += '--inference_iter %d ' % best_cpkt
            cmd += '--save_seg_dir ./output_best '
            print('run inference for training data for the best iteration (%d)...' % best_cpkt)
            os.system(cmd)
        else:
            print('Skip inference for the training data of the best model')
        # copy validation data inference output to output_best
        cmd = 'rsync -vah --progress %s/*.nii.gz %s' % \
            (os.path.join(model_dir, 'output_iter%d' % best_cpkt), output_dir)
        os.system(cmd)
        # run evaluation for training + evaluation data
        assert os.path.exists(output_dir), 'Something went wrong with the prediction of the best iter (%d)' % best_cpkt
        cmd = '%s/brats19/run_evaluation.py ' % NIFTYNET_PATH
        cmd += '--output_dir %s ' % output_dir
        cmd += '--data_split %s ' % data_split_file
        cmd += '--data_to_infer all '
        print('run validation evaluation for all data for the best iteration (%d)...' % best_cpkt)
        os.system(cmd)

    print('')
    best_model_path = os.path.join(model_dir, 'best_model.txt')
    with open(best_model_path, 'w') as f:
        line = 'the best model corresponds to iteration %d (mean dice=%f)' % (best_cpkt, best_mean_dice)
        print(line)
        f.write(line + '\n')
        for label in ['whole_tumor', 'core_tumor', 'enhancing_tumor']:
            line = 'label: {}, (validation) Dice mean: {}+/-{}'.format(label,
                                                     round(100*np.mean(best_scores[label]), 1),
                                                     round(100*np.std(best_scores[label]), 1))
            print(line)
            f.write(line + '\n')
    print('')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_dir',
                        help='folder containing the prediction to evaluate',
                        default=os.path.join(NIFTYNET_PATH, 'models_JADE', 'BraTS19', 'u_mvae'))
    parser.add_argument('--data_split',
                        help='dataset split csv to use',
                        default=os.path.join(os.path.dirname(PREPROCESSED_BRATS_PATH), 'dataset_split_fold0.csv'))
    args = parser.parse_args()
    model_dir = args.model_dir
    data_split_file = args.data_split

    main(model_dir, data_split_file)
