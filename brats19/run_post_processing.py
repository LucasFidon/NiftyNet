import os
import numpy as np
import pandas as pd
import nibabel as nib
from argparse import ArgumentParser
from definitions import ORIGINAL_VALIDATION_BRATS_PATH, NIFTYNET_PATH
from run_crop_volumes import crop_zeros, load_scans_BRATS, OUTPUT_AFFINE


def decrop(scalar_img, ori_pat_folder, pad_value=0):
    # load original T1, T1c, T2 and Flair
    ori_img_data, _ = load_scans_BRATS(ori_pat_folder)
    # get the coordinates that were used for the crop
    x_, _x, y_, _y, z_, _z = crop_zeros(ori_img_data)
    # create the de-cropped scalar image
    decrop_img = np.zeros_like(ori_img_data[::,::,::,0])
    if pad_value != 0:
        decrop_img.fill(pad_value)
    decrop_img[x_:_x, y_:_y, z_:_z] = scalar_img
    return decrop_img


def convert_labels(seg):
    seg[seg==3] = 4


def load_output_proba(pat_id, output_dir):
    proba_path = os.path.join(output_dir, 'window_seg_%s_.nii.gz' % pat_id)
    assert os.path.exists(proba_path), "Cannot found %s" % proba_path
    proba_img = np.squeeze(nib.load(proba_path).get_data())
    return proba_img


def main(proba_output_dir, data_split_csv):
    save_folder = '%s_postprocessed' % proba_output_dir
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # get list of inference patient id
    data_split = pd.read_csv(data_split_csv, header=None)
    pat_id_list = data_split[0].tolist()
    num_pat = len(pat_id_list)
    i = 0
    # post-process the output for each patient
    for pat_id in pat_id_list:
        # load proba images
        proba_img = load_output_proba(pat_id, proba_output_dir)
        # merge/argmax of the proba images
        seg = np.argmax(proba_img, axis=3)
        # decrop the segmentation
        ori_pat_folder = os.path.join(ORIGINAL_VALIDATION_BRATS_PATH, pat_id)
        decrop_seg = decrop(seg, ori_pat_folder)
        convert_labels(decrop_seg)
        # save the segmentation
        save_path = os.path.join(save_folder, '%s.nii.gz' % pat_id)
        decrop_seg_nii = nib.Nifti1Image(decrop_seg, OUTPUT_AFFINE)
        nib.save(decrop_seg_nii, save_path)
        i += 1
        print('')
        print('%d / %d' % (i, num_pat))
        print("%s has been post processed and saved at %s" % (pat_id, save_path))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_dir',
                        default=os.path.join(NIFTYNET_PATH, 'models_JADE',
                                             'BraTS19', 'u_mvae_wass_focal_long'))
    args = parser.parse_args()
    model_dir = args.model_dir
    proba_output_dir = os.path.join(model_dir, 'output_validation')
    data_split_csv = os.path.join(os.path.dirname(ORIGINAL_VALIDATION_BRATS_PATH), 'dataset_split_valid.csv')
    main(proba_output_dir, data_split_csv)
