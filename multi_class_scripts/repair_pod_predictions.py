from os import environ, listdir
from os.path import join
import nibabel as nib
from tqdm import tqdm


predp = join(environ['OV_DATA_BASE'], 'predictions', 'OV04', 'pod_067',
             'larger_res_encoder', 'cross_validation')

for case in tqdm(listdir(predp)):
    
    img = nib.load(join(predp, case))
    seg = img.get_fdata()
    seg_new = 9 * (seg > 0).astype(int)
    img_new = nib.Nifti1Image(seg_new.astype(int), img.affine, img.header)
    nib.save(img_new, join(predp, case))
