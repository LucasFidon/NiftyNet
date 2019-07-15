import os
import random
import csv
import nibabel
import numpy as np
from scipy import ndimage


DATASET_PATH_LABEL = '/home/lucasf/data/fetalMR_Guotai/label'
DATASET_PATH_IMAGE = '/home/lucasf/data/fetalMR_Guotai/image'

SAVE_PATH_LABEL = '/home/lucasf/data/fetalMR_Guotai/label_slices'
SAVE_PATH_IMAGE = '/home/lucasf/data/fetalMR_Guotai/image_slices'

DATA_SPLIT = {'validation': 0.1, 'inference': 0.2}
assert DATA_SPLIT['validation'] + DATA_SPLIT['inference'] < 1.
DATA_SPLIT['training'] = 1. - DATA_SPLIT['validation'] - DATA_SPLIT['inference']

SIZE = 192.  # in Guotai paper: 96


def split_and_save_slices(file_path, save_folder, is_label=False):
    slice_name_list = []
    slice_name_more_than_1_label = []
    img = nibabel.load(file_path).get_data()
    n_stack = np.min(img.shape)
    # unstack the slices
    slices = np.dsplit(img.transpose(np.argsort(img.shape)[::-1]), n_stack)
    # save the slices
    slice_n = 1
    for s in slices:
        save_name = os.path.basename(file_path).replace('.nii', 'slice_%d.nii' % slice_n)
        save_path = os.path.join(save_folder, save_name)
        if is_label:  ## keep only the label for the brain
        #     s[s == 1] = 0
            s[s > 2] = 0  # set maternal kidney and fetal lunges as background when present
        #     s[s == 2] = 1
            if len(np.unique(s)) > 1:  # more than 1 label in the slice
                slice_name_more_than_1_label.append(save_path)
        s_resized = resize_image(s, is_label)
        slice_nii = nibabel.Nifti1Image(s_resized, np.eye(4))
        nibabel.save(slice_nii, save_path)
        slice_name_list.append(save_path)
        slice_n += 1
    return slice_name_list, slice_name_more_than_1_label


def get_pat_id_from_path(img_path):
    to_remove = ['.nii', '.gz', '_seg']
    pat_id = os.path.basename(img_path)
    for w in to_remove:
        pat_id = pat_id.replace(w, '')
    return pat_id


def resize_image(full_res_img, is_label):
    zoom = (SIZE / full_res_img.shape[0], SIZE / full_res_img.shape[1], 1)
    if is_label:
        return ndimage.zoom(full_res_img, zoom, order=0)
    else:
        return ndimage.zoom(full_res_img, zoom, order=1)


if __name__ == '__main__':
    image_dict = {}  # slice id -> slice image path
    label_dict = {}  # slice id -> slice label path
    data_split_dict = {}  # slice id -> partition (validation, training or inference)
    # we keep track of the slices with more than 1 label in case we want to use only them for training
    more_than_1_label = []  # list of slice id with more than 1 label (i.e. not just background)

    # prepare the partition of the patient into validation - training - inference
    n_pat = len([n for n in os.listdir(DATASET_PATH_IMAGE)])
    n_validation = round(DATA_SPLIT['validation'] * n_pat)
    n_inference = round(DATA_SPLIT['inference'] * n_pat)
    n_training = n_pat - n_validation - n_inference
    partition = ['training']*n_training + ['validation']*n_validation + ['inference']*n_inference
    random.shuffle(partition)

    # create the slices dataset
    if not(os.path.exists(SAVE_PATH_IMAGE)):
        os.mkdir(SAVE_PATH_IMAGE)
    partition_index = 0
    assert os.path.exists(DATASET_PATH_IMAGE)
    for f_n in os.listdir(DATASET_PATH_IMAGE):
        f_path = os.path.join(DATASET_PATH_IMAGE, f_n)
        slice_list, _ = split_and_save_slices(f_path, SAVE_PATH_IMAGE)
        for slice_path in slice_list:
            slice_id = get_pat_id_from_path(slice_path)
            image_dict[slice_id] = slice_path
            data_split_dict[slice_id] = partition[partition_index]
        partition_index += 1

    if not(os.path.exists(SAVE_PATH_LABEL)):
        os.mkdir(SAVE_PATH_LABEL)
    assert os.path.exists(DATASET_PATH_LABEL)
    for f_n in os.listdir(DATASET_PATH_LABEL):
        f_path = os.path.join(DATASET_PATH_LABEL, f_n)
        slice_list, slice_more_than_1_label = split_and_save_slices(f_path, SAVE_PATH_LABEL, is_label=True)
        for slice_path in slice_list:
            slice_id = get_pat_id_from_path(slice_path)
            label_dict[slice_id] = slice_path
            if slice_path in slice_more_than_1_label:
                more_than_1_label.append(slice_id)

    # save csv files for all slices path
    image_csv_path = os.path.join(os.path.dirname(SAVE_PATH_IMAGE), 'image.csv')
    with open(image_csv_path, mode='w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for slice_id in list(image_dict.keys()):
            writer.writerow([slice_id, image_dict[slice_id]])

    label_csv_path = os.path.join(os.path.dirname(SAVE_PATH_IMAGE), 'label.csv')
    with open(label_csv_path, mode='w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for slice_id in list(label_dict.keys()):
            writer.writerow([slice_id, label_dict[slice_id]])

    # save csv files for the subset of slices with more than 1 label
    image_csv_path = os.path.join(os.path.dirname(SAVE_PATH_IMAGE), 'image_more_than_1_label.csv')
    with open(image_csv_path, mode='w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for slice_id in more_than_1_label:
            writer.writerow([slice_id, image_dict[slice_id]])

    label_csv_path = os.path.join(os.path.dirname(SAVE_PATH_IMAGE), 'label_more_than_1_label.csv')
    with open(label_csv_path, mode='w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for slice_id in more_than_1_label:
            writer.writerow([slice_id, label_dict[slice_id]])

    # save csv with the slit training/validation/inference for all the slices
    split_csv_path = os.path.join(os.path.dirname(SAVE_PATH_IMAGE), 'dataset_split.csv')
    with open(split_csv_path, mode='w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for slice_id in list(data_split_dict.keys()):
            writer.writerow([slice_id, data_split_dict[slice_id]])
