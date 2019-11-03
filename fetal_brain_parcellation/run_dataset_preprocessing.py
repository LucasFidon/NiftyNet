import os
import random
import csv
import nibabel as nib
import numpy as np
from definitions import LABELS, BOSTON2OURS, LABELS_BOSTON
from scipy import ndimage


DATASET_PATH = os.path.join(os.environ['HOME'], 'data',
                            'fetal_scans_and_segmentations',
                            'SRR_and_Segmentations')
BOSTON_DATASET_PATH = os.path.join(os.environ['HOME'], 'data',
                                   'fetal_brain_atlases',
                                   'Gholipour2017_atlas_NiftyMIC')
PREPROCESSED_DATASET_PATH =  os.path.join(os.environ['HOME'], 'data',
                            'fetal_scans_and_segmentations',
                            'SRR_and_Segmentations_preprocessed')

RUN_PREPROCESSING = False  # only for fetal brain data from Nada
USE_BOSTON_DATA = False
# RUN_PREPROCESSING_BOSTON = False

DATA_SPLIT = {'validation': 0.4, 'inference': 0.}
assert DATA_SPLIT['validation'] + DATA_SPLIT['inference'] < 1.
DATA_SPLIT['training'] = 1. - DATA_SPLIT['validation'] - DATA_SPLIT['inference']

BORDER_MASK = 15
MIN_IMG_SIZE = 64

def load_data(img_path):
    img = np.squeeze(nib.load(img_path).get_data())
    return img

def get_affine(img_path):
    aff = nib.load(img_path).affine
    return aff

def get_study_name_from_file_path(file_path):
    study_name = os.path.basename(os.path.dirname(file_path))
    return study_name

def normalize_img_size(i_start, i_end):
    m = i_end - i_start - MIN_IMG_SIZE
    n_i_start = i_start
    n_i_end = i_end
    if m < 0:
        n_i_start = max(i_start - m // 2, 0)
        n_i_end = n_i_start + MIN_IMG_SIZE
    return n_i_start, n_i_end

def crop_around_mask(scan_path, mask_path):
    # crop 3d scan around the mask of the brain
    scan = load_data(scan_path)
    mask = load_data(mask_path)
    # get the coordinates of the voxels inside the mask
    i_nz, j_nz, k_nz = np.nonzero(mask)
    i_start = max(0, np.min(i_nz) - BORDER_MASK)
    j_start = max(0, np.min(j_nz) - BORDER_MASK)
    k_start = max(0, np.min(k_nz) - BORDER_MASK)
    i_end = min(scan.shape[0], np.max(i_nz) + BORDER_MASK)
    j_end = min(scan.shape[1], np.max(j_nz) + BORDER_MASK)
    k_end = min(scan.shape[2], np.max(k_nz) + BORDER_MASK)
    # normalise the coordinates
    i_s, i_e = normalize_img_size(i_start, i_end)
    j_s, j_e = normalize_img_size(j_start, j_end)
    k_s, k_e = normalize_img_size(k_start, k_end)
    # crop the scan and the mask
    cropped_scan = scan[i_s:i_e, j_s:j_e, k_s:k_e]
    cropped_mask = mask[i_s:i_e, j_s:j_e, k_s:k_e]
    # save cropped scan and mask
    study_n = get_study_name_from_file_path(scan_path)
    save_dir_path = os.path.join(PREPROCESSED_DATASET_PATH, study_n)
    if not(os.path.exists(save_dir_path)):
        os.mkdir(save_dir_path)
    cropped_scan_nii = nib.Nifti1Image(cropped_scan, get_affine(scan_path))
    cropped_mask_nii = nib.Nifti1Image(cropped_mask, get_affine(mask_path))
    scan_p = os.path.join(save_dir_path, '%s_img.nii.gz' % study_n)
    mask_p = os.path.join(save_dir_path, '%s_seg.nii.gz' % study_n)
    nib.save(cropped_scan_nii, scan_p)
    nib.save(cropped_mask_nii, mask_p)
    return study_n, scan_p, mask_p

def merge_labels(study_folder_path):
    # put all the segmentation in one nii file:
    # Labels:
    # 0: background
    # 1: white matter
    # 2: Ventricules (part of CSF)
    # 3: Cerebellum
    # 4: Other brain tissues

    # load the segmentations
    seg = {}
    aff = np.eye(4)
    for file_n in os.listdir(study_folder_path):
        if '.nii' in file_n:
            seg_p = os.path.join(study_folder_path, file_n)
            if 'mask' in file_n:
                seg['other_brain'] = load_data(seg_p)
                aff = get_affine(seg_p)
            elif 'wm' in file_n:
                seg['wm'] = load_data(seg_p)
            elif 'csf' in file_n:
                seg['ventricules'] = load_data(seg_p)
            elif 'cerebellum' in file_n:
                seg['cerebellum'] = load_data(seg_p)
    # create the unified segmentation
    seg_img = np.zeros_like(seg['other_brain'])
    seg_img[seg['other_brain'] > 0] = LABELS['other_brain']
    seg_img[seg['wm'] > 0] = LABELS['wm']
    seg_img[seg['ventricules'] > 0] = LABELS['ventricules']
    seg_img[seg['cerebellum'] > 0] = LABELS['cerebellum']
    # print(study_folder_path)
    # print(seg_img.shape)
    # print('')
    # save the unified segmentation
    # save_dir_path = os.path.join(PREPROCESSED_DATASET_PATH, os.path.basename)
    # if not(os.path.exists(save_dir_path)):
    #     os.mkdir(save_dir_path)
    seg_nii = nib.Nifti1Image(seg_img, aff)
    nib.save(seg_nii, os.path.join(study_folder_path, 'seg.nii.gz'))

def merge_labels_Boston(seg_path, csf_erosion=15):
    # load volumes
    study_name = os.path.basename(seg_path).split('_')[0]
    seg_Boston_labels = load_data(seg_path)
    img = load_data(os.path.join(os.path.dirname(seg_path), '%s.nii.gz' % study_name))
    aff = get_affine(os.path.join(os.path.dirname(seg_path), '%s.nii.gz' % study_name))

    # initialise the segmentation for our labels
    seg = np.zeros_like(seg_Boston_labels)

    # set all the foreground to "other_brain" for initialization
    seg[img > 300] = 4

    # convert labels into our convention
    for ori_l in list(BOSTON2OURS.keys()):
        seg[seg_Boston_labels == ori_l] = BOSTON2OURS[ori_l]

    # fix the difference of convention for the ventricules:
    # third and fourth ventricules are included in CSF in the Boston labels
    seg_foreground = np.copy(seg)
    seg_foreground[seg > 300] = 1
    # erode the foreground mask
    seg_for_eroded = ndimage.morphology.binary_erosion(seg_foreground,
                                                       iterations=csf_erosion).astype(seg.dtype)
    # 3rd and 4th ventricules are approximately the CSF inside the eroded foreground mask
    seg[np.logical_and(seg_Boston_labels == LABELS_BOSTON['CSF'], seg_for_eroded)] = 2

    # BOSTON will refer to the common name of the "average" patient from the Boston atlas
    save_folder = os.path.join(DATASET_PATH, 'BOSTON_%s' % study_name)

    img_save_path = os.path.join(save_folder, 'srr.nii.gz')
    seg_save_path = os.path.join(save_folder, 'seg.nii.gz')
    seg_nii = nib.Nifti1Image(seg, aff)
    img_nii = nib.Nifti1Image(img, aff)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    nib.save(seg_nii, seg_save_path)
    nib.save(img_nii, img_save_path)
    return img_save_path, seg_save_path

if __name__ == '__main__':
    pat_id = []
    img_dict = {}  # study id -> img path
    seg_dict = {}  # study id -> seg path
    data_split_dict = {}  # slice id -> partition (validation, training or inference)
    split_pat = {}  # it is necessary to split based on the patients and not based on the studies

    if not(os.path.exists(PREPROCESSED_DATASET_PATH)):
        os.mkdir(PREPROCESSED_DATASET_PATH)

    # preprocess data
    if RUN_PREPROCESSING:
        for study_n in os.listdir(DATASET_PATH):
            if not('.' in study_n):
                study_p = os.path.join(DATASET_PATH, study_n)
                merge_labels(study_p)
                seg_p = os.path.join(study_p, 'seg.nii.gz')
                img_p = ''
                for file_n in os.listdir(study_p):
                    if 'srr' in file_n and not('mask' in file_n):
                        img_p = os.path.join(study_p, file_n)
                study_n, img_p, seg_p = crop_around_mask(img_p, seg_p)
                # get the patient id from the study id
                pat_n = study_n.split('_')[0]
                if not pat_n in pat_id:
                    pat_id.append(pat_n)
                img_dict[study_n] = img_p
                seg_dict[study_n] = seg_p

    if USE_BOSTON_DATA:
        # preprocess Boston fetal brain atlas dataset
        # for w in range(21, 31):
        for w in range(21, 38):
            study_n = 'STA%d' % w
            ori_seg_path = os.path.join(BOSTON_DATASET_PATH, '%s_WMZparc.nii.gz' % study_n)
            if not(os.path.exists(ori_seg_path)):
                ori_seg_path = os.path.join(BOSTON_DATASET_PATH,
                                            '%s_parc.nii.gz' % study_n)
            # merge labels and moves seg and img to a new study folder
            img_p, seg_p = merge_labels_Boston(ori_seg_path, csf_erosion=w-5)
            # crop seg and img around the foreground
            study_n, img_p, seg_p = crop_around_mask(img_p, seg_p)
            # get the patient id from the study id
            pat_n = study_n.split('_')[0]
            # if not pat_n in pat_id:
            #     pat_id.append(pat_n)
            img_dict[study_n] = img_p
            seg_dict[study_n] = seg_p
        # atlas data are always in training
        split_pat[pat_n] = 'training'


    # split data into training/validation/inference
    random.shuffle(pat_id)
    n_pat = len(pat_id)
    n_validation = round(DATA_SPLIT['validation'] * n_pat)
    n_inference = round(DATA_SPLIT['inference'] * n_pat)
    n_training = n_pat - n_validation - n_inference

    for i in range(n_training):
        split_pat[pat_id[i]] = 'training'
    for i in range(n_training, n_training+n_validation):
        split_pat[pat_id[i]] = 'validation'
    for i in range(n_training+n_validation, n_pat):
        split_pat[pat_id[i]] = 'inference'

    for study_name in list(img_dict.keys()):
        pat_name = study_name.split('_')[0]
        data_split_dict[study_name] = split_pat[pat_name]

    # save csv files for images and segmentations path
    image_csv_path = os.path.join(PREPROCESSED_DATASET_PATH, 'image.csv')
    with open(image_csv_path, mode='w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        for slice_id in list(img_dict.keys()):
            writer.writerow([slice_id, img_dict[slice_id]])

    label_csv_path = os.path.join(PREPROCESSED_DATASET_PATH, 'label.csv')
    with open(label_csv_path, mode='w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        for slice_id in list(seg_dict.keys()):
            writer.writerow([slice_id, seg_dict[slice_id]])

    # save csv with the slit training/validation/inference for all the slices
    split_csv_path = os.path.join(PREPROCESSED_DATASET_PATH, 'dataset_split.csv')
    with open(split_csv_path, mode='w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        for slice_id in list(data_split_dict.keys()):
            writer.writerow([slice_id, data_split_dict[slice_id]])
