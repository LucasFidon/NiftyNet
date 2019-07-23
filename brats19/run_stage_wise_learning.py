import os
from definitions import *


if __name__ == "__main__":
   learning_rate = 0.001
   start_from_iter = 0
   learning_rate_decay = 0.25
   n_iter_per_stage = 10000
   for i in range(9):
       print('-----------------------------------')
       print('STAGE %d of the stage-wise learning' % (i+1))
       print('-----------------------------------')
       cmd = 'python net_run.py train'
       cmd += ' -a niftynet.extension.variational_segmentation_application.VariationalSegmentationApplication'
       cmd += ' -c brats19/config.ini'
       # add options
       cmd += ' --starting_iter %d' % start_from_iter
       cmd += ' --max_iter %d' % n_iter_per_stage
       cmd += ' --lr %f' % learning_rate
       os.system(cmd)

       # update hyperparameters for next stage
       learning_rate *= learning_rate_decay
       start_from_iter += n_iter_per_stage
       print('')
