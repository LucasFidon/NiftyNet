import os


# PATHS
# ROOT_FOLDER = os.environ['HOME']
ROOT_FOLDER = '/'
NIFTYNET_PATH = os.path.join(ROOT_FOLDER, 'workspace', 'NiftyNet')
ORIGINAL_BRATS_PATH = os.path.join(
    ROOT_FOLDER, 'data', 'BraTS2019', 'MICCAI_BraTS_2019_Data_Training')
PREPROCESSED_BRATS_PATH = os.path.join(
    ROOT_FOLDER, 'data', 'BraTS2019', 'MICCAI_BraTS_2019_Data_Training_crop')

# HYPERPARAMETERS
MODALITY_NAMES = ['Flair', 'T1', 'T1c', 'T2']
NUMBER_OF_FOLDS = 5
MIN_IMAGE_SIZE = 140

def check_path(path):
    msg = '%s not Found. Please your path in NiftyNet/brats19/definitions.py' % path
    assert os.path.exists(path), msg

def check_all_path_definition():
    check_path(ROOT_FOLDER)
    check_path(NIFTYNET_PATH)
    # check_path(ORIGINAL_BRATS_PATH)  # does not have to exist on the cluster...
    check_path(PREPROCESSED_BRATS_PATH)
