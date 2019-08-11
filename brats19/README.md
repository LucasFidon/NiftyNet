## Robust ensemble of variational autoencoders for brain tumor segmentation and uncertainty quantification

The first time you use this code you need to:
1. download the BRATS 2019 training dataset
2. change the path in definitions.py according to your local paths
3. run "python run_crop_volumes.py"
4. run "python run_split_subjects.py"

You are now ready to train the robust ensemble of VAEs for brain tumor segmentation.
Simply run:
"python run_pipeline.py"