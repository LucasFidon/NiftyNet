"""
This script renames BRATS dataset to OUTPUT_path,
each subject's images will be cropped and renamed to
"TYPEindex_modality.nii.gz".

output dataset folder will be created if not exists, and content
in the created folder will be, for example:

OUTPUT_path:
   HGG100_Flair.nii.gz
   HGG100_Label.nii.gz
   HGG100_T1c.nii.gz
   HGG100_T1.nii.gz
   HGG100_T2.nii.gz
   ...

Each .nii.gz file in OUTPUT_path will be cropped with a tight bounding box
using function crop_zeros defined in this script.

Please change BRATS_path and OUTPUT_path accordingly to the preferred folder
"""
import os

import SimpleITK as sitk
import nibabel
import numpy as np
from definitions import *

# change here to the directory of downloaded BRATS data')
# BRATS_path = os.path.join(
#     os.environ['HOME'],'data', 'BraTS2019', 'MICCAI_BraTS_2019_Data_Training')
BRATS_path = ORIGINAL_BRATS_PATH

# change here to the directory of preferred output directory')
# OUTPUT_path = os.path.join(
#     os.environ['HOME'],'data', 'BraTS2019', 'MICCAI_BraTS_2019_Data_Training_crop')
OUTPUT_path = PREPROCESSED_BRATS_PATH

# Aff to use with BRATS dataset
OUTPUT_AFFINE = np.array(
    [[-1, 0, 0, 0],
     [0, -1, 0, 239],
     [0, 0, 1, 0],
     [0, 0, 0, 1]])

mod_names17 = ['flair', 't1', 't1ce', 't2']
mod_names15 = ['Flair', 'T1', 'T1c', 'T2']

labels=[1,2,4]

def normalize_img_size(xmin, xmax, x_dim):
    if xmax-xmin < MIN_IMAGE_SIZE:
        ecart = int((MIN_IMAGE_SIZE - (xmax-xmin))/2)
        xmin = max(0, xmin - ecart)
        xmax = min(x_dim, xmin + MIN_IMAGE_SIZE)
        # xmax = xmax+ecart+1
        # xmin = xmin-ecart
    return xmin,xmax
           

def replace_values_seg_zone(seg):
    zero = (seg==0).astype(float)
    wt = (seg>0).astype(float)
    ct = (seg==1).astype(float) + (seg==4).astype(float)
    en = (seg==4).astype(float)
    labels = np.stack([zero,wt,ct,en],axis=-1)
    return labels


def replace_values_seg(img_array):
    if np.sum(img_array == 3)>0:
        print('okay pototototot')
    for k in range(3):
        img_array[img_array == labels[k]]= k+1
    return img_array


def crop_zeros(img_array):
    if len(img_array.shape) == 4:
        img_array = np.amax(img_array, axis=3)
    assert len(img_array.shape) == 3
    x_dim, y_dim, z_dim = tuple(img_array.shape)
    x_zeros, y_zeros, z_zeros = np.where(img_array == 0.)
    # x-plans that are not uniformly equal to zeros
    
    try:
        x_to_keep, = np.where(np.bincount(x_zeros) < y_dim * z_dim)
        x_min = min(x_to_keep)
        x_max = max(x_to_keep) + 1
    except Exception :
        x_min = 0
        x_max = x_dim
    try:
        y_to_keep, = np.where(np.bincount(y_zeros) < x_dim * z_dim)
        y_min = min(y_to_keep)
        y_max = max(y_to_keep) + 1
    except Exception :
        y_min = 0
        y_max = y_dim
    try :
        z_to_keep, = np.where(np.bincount(z_zeros) < x_dim * y_dim)
        z_min = min(z_to_keep)
        z_max = max(z_to_keep) + 1
    except:
        z_min = 0
        z_max = z_dim

    x_min, x_max = normalize_img_size(x_min, x_max, x_dim)
    y_min, y_max = normalize_img_size(y_min, y_max, y_dim)
    z_min, z_max = normalize_img_size(z_min, z_max, z_dim)

    return x_min, x_max, y_min, y_max, z_min, z_max


def pad_volume(vol_array):
    require_padding = False
    # pas on both sides with zeros if the minimal size is not reached
    pad_x = 0
    if vol_array.shape[0] < MIN_IMAGE_SIZE:
        pad_x = (MIN_IMAGE_SIZE - vol_array.shape[0]) // 2 + 1
        require_padding = True
    pad_y = 0
    if vol_array.shape[1] < MIN_IMAGE_SIZE:
        pad_y = (MIN_IMAGE_SIZE - vol_array.shape[1]) // 2 + 1
        require_padding = True
    pad_z = 0
    if vol_array.shape[2] < MIN_IMAGE_SIZE:
        pad_z = (MIN_IMAGE_SIZE - vol_array.shape[2]) // 2 + 1
        require_padding = True
    if require_padding:
        print('the volume is padded with zero to fulfil the minimum image size')
        if len(vol_array.shape) == 3:
            pad_vol = np.pad(vol_array,
                             ((pad_x, pad_x), (pad_y, pad_y), (pad_z, pad_z)),
                             'constant')
        else:
            pad_vol = np.pad(vol_array,
                             ((pad_x, pad_x), (pad_y, pad_y), (pad_z, pad_z), (0,0)),
                             'constant')
        return pad_vol
    else:
        return vol_array


def load_scans_BRATS(pat_folder, with_seg=False):
    nii_fnames = [f_name for f_name in os.listdir(pat_folder)
                  if f_name.endswith(('.nii', '.nii.gz')) 
                  and not f_name.startswith('._')]
    img_data = []
    for mod_n in mod_names17:
        file_n = [f_n for f_n in nii_fnames if (mod_n + '.') in f_n][0]
        mod_data = nibabel.load(os.path.join(pat_folder, file_n)).get_data()
        img_data.append(mod_data)
    img_data = np.stack(img_data, axis=-1)
    if not with_seg:
        return img_data, None
    else:
        file_n = [f_n for f_n in nii_fnames if ('seg.') in f_n][0]
        seg_data = nibabel.load(os.path.join(pat_folder, file_n)).get_data()
        return img_data, seg_data


def save_scans_BRATS(pat_name, img_data, seg_data=None):
    # save_mod_names = ['Flair', 'T1', 'T1c', 'T2']
    save_mod_names = MODALITY_NAMES
    save_seg_name = 'Label'
    assert img_data.shape[3] == 4
    for mod_i in range(len(save_mod_names)):
        save_name = '%s_%s.nii.gz' % (pat_name, save_mod_names[mod_i])
        save_path = os.path.join(OUTPUT_path, save_name)
        mod_data_nii = nibabel.Nifti1Image(img_data[:, :, :, mod_i],
                                           OUTPUT_AFFINE)
        nibabel.save(mod_data_nii, save_path)
    print('saved to {}'.format(OUTPUT_path))
    if seg_data is not None:
        save_name = '%s_%s.nii.gz' % (pat_name, save_seg_name)
        save_path = os.path.join(OUTPUT_path, save_name)
        seg_data_nii = nibabel.Nifti1Image(seg_data, OUTPUT_AFFINE)
        nibabel.save(seg_data_nii, save_path)


def main(pat_category_list=('HGG', 'LGG'), crop=False):
    min_s_ori = 500
    min_s_crop = 500
    for pat_cat in pat_category_list:
        pat_ID = 0
        for pat_folder_name in os.listdir(os.path.join(BRATS_path, pat_cat)):
            pat_ID += 1
            # Load
            pat_folder = os.path.join(BRATS_path, pat_cat, pat_folder_name)
            try:
                print(pat_folder)
                img_data, seg_data = load_scans_BRATS(
                    pat_folder, with_seg=True)
            except OSError:
                print('skipping %s' % pat_folder)
                continue
                pass
            print("subject: {}, shape: {}".format(pat_folder, img_data.shape))
            for s in list(img_data.shape)[:-1]:
                if s < min_s_ori:
                    min_s_ori = s
            # Cropping
            if crop:
                x_, _x, y_, _y, z_, _z = crop_zeros(img_data)
                img_data = pad_volume(img_data[x_:_x, y_:_y, z_:_z, :])
                seg_data = pad_volume(seg_data[x_:_x, y_:_y, z_:_z])
                seg_data = replace_values_seg(seg_data)
                print('shape cropping: {}'.format(img_data.shape))
                for s in list(img_data.shape)[:-1]:
                    if s < min_s_crop:
                        min_s_crop = s
            # Save with name convention
            pat_name = '%s%d' % (pat_cat, pat_ID)
            # remove '_' from pat_name to match name convention
            pat_name = pat_name.replace('_', '')
            save_scans_BRATS(pat_name, img_data, seg_data)
            print('')
            print('min size before cropping: %d' % min_s_ori)
            print('min size after cropping: %d' % min_s_crop)


if __name__ == '__main__':
    if not os.path.exists(BRATS_path):
        raise ValueError(
            'please change "BRATS_path" in this script to '
            'the BRATS17 challenge dataset. '
            'Dataset not found: {}'.format(BRATS_path))
    if not os.path.exists(OUTPUT_path):
        os.makedirs(OUTPUT_path)
    main(crop=True)
    # main(['HGG'], dataset='BRATS15', crop=False)
