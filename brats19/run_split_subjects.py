import os
import csv
from definitions import *


def get_all_patient_id(pat_folder):
    id_list = []
    for file_name in os.listdir(pat_folder):
        if '.nii' in file_name:
            id = '%s_' % (file_name.split('_'))[0]
            if not id in id_list:
                id_list.append(id)
    return id_list

def split_id_into_HGG_and_LGG(id_list):
    LGG_list = []
    HGG_list = []
    for id in id_list:
        if 'HGG' in id:
            HGG_list.append(id)
        else:
            assert ('LGG' in id), 'cannot assess if %s is an HGG or a LGG' % id
            LGG_list.append(id)
    return HGG_list, LGG_list

def split_HGG_and_LGG_into_folds(HGG_list, LGG_list):
    # make sure HGG and LGG patients are distributed across all folds
    data_fold = {}  # map patient id -> fold number
    for i in range(len(HGG_list)):
        data_fold[HGG_list[i]] = i % NUMBER_OF_FOLDS
    for i in range(len(LGG_list)):
        data_fold[LGG_list[i]] = (-i) % NUMBER_OF_FOLDS
    return data_fold


if __name__ == '__main__':
    id_list = get_all_patient_id(PREPROCESSED_BRATS_PATH)
    HGG_list, LGG_list = split_id_into_HGG_and_LGG(id_list)
    # map patient id -> fold number
    data_fold = split_HGG_and_LGG_into_folds(HGG_list, LGG_list)
    # save the csv files for the different folds
    for fold in range(NUMBER_OF_FOLDS):
        split_csv_path = os.path.join(PREPROCESSED_BRATS_PATH, 'dataset_split_fold%s.csv' % fold)
        with open(split_csv_path, mode='w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
            for pat_id in list(data_fold.keys()):
                if data_fold[pat_id] == fold:
                    writer.writerow([pat_id, 'validation'])
                else:
                    writer.writerow([pat_id, 'inference'])
