import os
import numpy as np
import pandas as pd
import csv
from argparse import ArgumentParser
from definitions import *

T = 20  # number of CNNs in the ensemble
SCORE_CSV_BASE = os.path.join(NIFTYNET_PATH, 'models', 'BraTS19')
NUM_SUBJECT_BRATS19 = 335


def compute_proba_from_score(t):
    model_path = os.path.join(SCORE_CSV_BASE, 'u_mvae_%d' % t)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    proba_csv_path = os.path.join(model_path, 'subject_proba_%d.csv' % t)
    if t == 0:
        uniform_proba_path = os.path.join(os.path.dirname(PREPROCESSED_BRATS_PATH),
                                          'subject_proba_0.csv')
        cmd = 'rsync -vah --progress %s %s' % (uniform_proba_path, proba_csv_path)
        os.system(cmd)
    else:
        # load scores of the previous CNNs
        scores_previous_cnn = []
        for i in range(t):
            scores_csv = os.path.join(SCORE_CSV_BASE, 'u_mvae_%d' % i, 'output_best', 'scores.csv')
            scores = pd.read_csv(scores_csv)
            assert len(scores['subject_id'].tolist()) == NUM_SUBJECT_BRATS19, "Please check file %s" % scores_csv
            scores_previous_cnn.append(scores)
        # compute the subject proba
        subject_x = {}
        for pat_id in scores_previous_cnn[0]['subject_id'].tolist():
            subject_x[pat_id] = 0
            for i in range(t):
                for roi in ['whole_tumor', 'core_tumor', 'enhancing_tumor']:
                    # we want to MINIMIZE the mean dice
                    subject_x[pat_id] -= scores_previous_cnn[i][roi][scores_previous_cnn[i]['subject_id']==pat_id].values[0]
        # compute the softmax with:
        # e ^ (x - max(x)) / sum(e^(x - max(x))
        # max(x) is substracted for numerical stabilities
        x = np.array([k for k in subject_x.values()])
        max_x = np.max(x)
        sum_e_x = np.sum(np.exp(x - max_x))
        # save the subject proba csv that will be used for training CNN #t
        with open(proba_csv_path, mode='w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
            for pat_id in scores_previous_cnn[0]['subject_id']:
                writer.writerow([pat_id, np.exp(subject_x[pat_id] - max_x) / sum_e_x])
        print('The subject probabilities csv %s has been created' % proba_csv_path)
    return proba_csv_path


def main(t0, start_iter_t0=0):
    for t in range(t0, T):
        print('\n--- COMPUTE SUBJECT PROBA FOR MODEL %d ---\n' % t)
        proba_csv_path = compute_proba_from_score(t)
        assert os.path.exists(proba_csv_path), \
            "Something went wrong with the subjects probability csv %s" % proba_csv_path

        print('\n--- TRAIN MODEL %d ---\n' % t)
        fold = t % NUMBER_OF_FOLDS
        start_iter = 0
        if t==t0:
            start_iter = start_iter_t0
        cmd = 'python %s/brats19/run_stage_wise_learning.py' % NIFTYNET_PATH
        cmd += ' --fold %d' % fold
        cmd += ' --cnn %d' % t
        cmd += ' --start_iter %d' % start_iter
        cmd += ' --subject_proba %s' % proba_csv_path
        os.system(cmd)

        print('\n--- RUN EARLY STOPPING FOR MODEL %d ---\n' % t)
        model_dir = os.path.join(SCORE_CSV_BASE, 'u_mvae_%d' %t)
        data_split_csv = os.path.join(os.path.dirname(PREPROCESSED_BRATS_PATH),
                                      'dataset_split_fold%d.csv' % fold)
        cmd = 'python %s/brats19/run_early_stopping.py' % NIFTYNET_PATH
        cmd += ' --model_dir %s' % model_dir
        cmd += ' --data_split %s' % data_split_csv
        os.system(cmd)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--start_cnn',
                        help='Number of CNNs already trained that must be skipped for training',
                        default=1)
    args = parser.parse_args()
    t0 = int(args.start_cnn)  # skip first CNNs (already trained)
    main(t0)
