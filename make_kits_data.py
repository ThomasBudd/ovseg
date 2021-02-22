from os import listdir, makedirs, environ
from os.path import join, exists
from shutil import copyfile
from ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
from tqdm import tqdm

kits_path = '/local/scratch/public/tb588/kits19/data'
ov_raw = join(environ['OV_DATA_BASE'], 'raw_data', 'kits19')

for subf in ['images', 'labels']:
    if not exists(join(ov_raw, subf)):
        makedirs(join(ov_raw, subf))

for case in tqdm(listdir(kits_path)):
    if exists(join(kits_path, case, 'segmentation.nii.gz')):
        copyfile(join(kits_path, case, 'segmentation.nii.gz'),
                 join(ov_raw, 'labels', 'case_{}.nii.gz'.format(case[-3:])))
        copyfile(join(kits_path, case, 'imaging.nii.gz'),
                 join(ov_raw, 'images', 'case_{}_0000.nii.gz'.format(case[-3:])))

# %% now preprocessing
raw_data = 'kits19'
preprocessing = SegmentationPreprocessing(use_only_classes=[9])
preprocessing.plan_preprocessing_raw_data(raw_data)
preprocessing.preprocess_raw_data(raw_data)
