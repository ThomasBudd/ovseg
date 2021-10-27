from os import environ, listdir
from os.path import join
import nibabel as nib
from tqdm import tqdm


predp = join(environ['OV_DATA_BASE'], 'predictions', 'OV04', 'pod_067',
             'larger_res_encoder', 'cross_validation')

for case in tqdm(listdir(predp)):
    
    img = nib.load(join(predp, case))
    seg = img.get_fdata()
    if seg.max() == 1:
        seg_new = seg * 9
        img_new = nib.Nifti1Image(arr, img.affine, img.header)
        nib.save(img_new, join(predp, case))
