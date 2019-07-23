import os


ORIGINAL_BRATS_PATH = os.path.join(
    os.environ['HOME'],'data', 'BraTS2019', 'MICCAI_BraTS_2019_Data_Training')

PREPROCESSED_BRATS_PATH = os.path.join(
    os.environ['HOME'],'data', 'BraTS2019', 'MICCAI_BraTS_2019_Data_Training_crop')

MODALITY_NAMES = ['Flair', 'T1', 'T1c', 'T2']

NUMBER_OF_FOLDS = 5