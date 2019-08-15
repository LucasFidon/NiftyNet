import os
from argparse import ArgumentParser
from definitions import NIFTYNET_PATH


def get_best_ckpt_iter(model_dir):
    best_model_txt = os.path.join(model_dir, 'best_model.txt')
    assert os.path.exists(best_model_txt), "Cannot find %s" % best_model_txt
    with open(best_model_txt, 'r') as f:
        content = f.read()
    words = content.split(' ')
    best_ckpt = int(words[6])
    return best_ckpt


def main(model_dir):
    assert os.path.exists(model_dir), 'Folder %s not found. Please check your --model_dir'
    best_ckpt = get_best_ckpt_iter(model_dir)

    # run inference
    cmd = '%s/net_run.py inference ' % NIFTYNET_PATH
    cmd += '-a niftynet.extension.variational_segmentation_application.VariationalSegmentationApplication '
    cmd += '-c %s/brats19/config_validation.ini ' % NIFTYNET_PATH
    cmd += '--model_dir %s ' % model_dir
    cmd += '--dataset_to_infer inference '
    cmd += '--inference_iter %d ' % best_ckpt
    cmd += '--save_seg_dir ./output_validation '
    os.system(cmd)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_dir',
                        default='/workspace/NiftyNet/models_JADE/BraTS19/u_mvae_wass_focal_long')
    args = parser.parse_args()
    model_dir = args.model_dir
    main(model_dir)
