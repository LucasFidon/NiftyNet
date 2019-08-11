import os
from definitions import *
from argparse import ArgumentParser


if __name__ == "__main__":
    # check that the paths hard-coded in definitions exist
    check_all_path_definition()

    parser = ArgumentParser()
    parser.add_argument('--fold',
                        help='fold number to use for splitting of the data into training/validation',
                        default=0)
    parser.add_argument('--cnn',
                        help='number of CNNs that have already been trained in the robust ensemble.',
                        default=0)
    parser.add_argument('--start_iter',
                        help='(re)start the training from this iteration',
                        default=0)
    parser.add_argument('--subject_proba',
                        help='Path to the csv containing the sampling proba for each subject.',
                        default=os.path.join(os.path.dirname(PREPROCESSED_BRATS_PATH),
                                                             'subject_proba_0.csv'))
    args = parser.parse_args()

    data_split = int(args.fold)
    cnn_number = int(args.cnn)  # number of already trained CNNs for the robust ensemble
    start_iter = int(args.start_iter)
    total_iter = 60000
    # total_iter = 20  # for quick test
    init_learning_rate = 0.001
    learning_rate_decay = 0.25
    n_iter_per_stage = 15000
    # n_iter_per_stage = 10  # for quick test
    save_every_n = 1000
    # save_every_n = 10  # for quick test

    # set path of the folder where the checkpoints will be saved
    model_dir = os.path.join(NIFTYNET_PATH, 'models', 'BraTS19', 'u_mvae_%d' % cnn_number)
    # model_dir = os.path.join(NIFTYNET_PATH, 'models', 'BraTS19', 'u_mvae_wass_focal_long')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # set path to csv to used for the splitting training/validation
    data_split_csv = os.path.join(os.path.dirname(PREPROCESSED_BRATS_PATH),
                                  'dataset_split_fold%d.csv' % data_split)
    assert os.path.exists(data_split_csv), ("the csv dataset_split_file %s cannot be found" % data_split_csv)

    # set path to the csv to use for the subject probabilities
    subject_proba_csv = args.subject_proba
    assert os.path.exists(subject_proba_csv), 'The csv of the sampling subject proba %s cannot be found' % subject_proba_csv

    # set the path to the config file to use
    config_path = os.path.join(NIFTYNET_PATH, 'brats19', 'config.ini')
    
    n_stages = total_iter // n_iter_per_stage
    learning_rates = [init_learning_rate*(learning_rate_decay**i) for i in range(n_stages)]

    stages_start_from_iter = [n_iter_per_stage*i for i in range(n_stages+1)]
    # we set the start_stage number to allow for resuming the training
    start_stage = start_iter // n_iter_per_stage
    stages_start_from_iter[start_stage] = start_iter

    for i in range(start_stage, n_stages):
       print('-----------------------------------')
       print('STAGE %d of the stage-wise learning' % (i+1))
       print('-----------------------------------')
       cmd = 'python net_run.py train'
       cmd += ' -a niftynet.extension.variational_segmentation_application.VariationalSegmentationApplication'
       cmd += ' -c %s' % config_path
       # add options
       cmd += ' --starting_iter %d' % stages_start_from_iter[i]
       cmd += ' --max_iter %d' % stages_start_from_iter[i+1]
       cmd += ' --lr %f' % learning_rates[i]
       cmd += ' --subject_proba_file %s' % subject_proba_csv
       cmd += ' --model_dir %s' % model_dir
       cmd += ' --save_every_n %d' % save_every_n
       os.system(cmd)
       print('')
