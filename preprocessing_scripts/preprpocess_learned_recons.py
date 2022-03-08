from ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
from os import listdir, environ, mkdir
from os.path import join, exists
from tqdm import tqdm
import nibabel as nib
import numpy as np

prepp = join(environ['OV_DATA_BASE'], 'preprocessed', 'OV04', 'pod_no_resizing')
predp = join(environ['OV_DATA_BASE'], 'predictions', 'OV04')

preprocessing = SegmentationPreprocessing()
preprocessing.load_preprocessing_parameters(prepp)

models = ['recon_fbp_convs_{}_eights_8_32'.format(i) for i in [1500, 2000, 2500]]

for model in models:
    print('converting '+model)

    if not exists(join(prepp, model)):
        mkdir(join(prepp, model))

    for folder in listdir(join(predp, model)):
        print(folder)
        print()

        for file in tqdm(listdir(join(predp, model, folder))):
            img = nib.load(join(predp, model, folder, file))
            im = img.get_fdata()
            sp = img.header['pixdim'][1:4]

            im_prep = preprocessing.preprocess_volume(im, sp)

            np.save(join(prepp, model, file[:-7]), im_prep.astype(np.float16))
