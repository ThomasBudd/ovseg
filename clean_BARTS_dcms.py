from ovseg.postprocessing.SegmentationPostprocessing import SegmentationPostprocessing
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from os import listdir, environ
from os.path import join, basename
from skimage.measure import label
from ovseg.utils.io import read_dcms, save_dcmrt_from_data_tpl
from tqdm import tqdm
from ovseg.utils.label_utils import reduce_classes

predp = join(environ['OV_DATA_BASE'], 'predictions', 'OV04', 'pod_om_08_25',
             'U-Net4_prg_lrn', 'BARTS_dcm_ensemble_5_6_7')

rawp = join(environ['OV_DATA_BASE'], 'raw_data', 'BARTS_dcm')

nii_files = [f for f in listdir(predp) if f.endswith('.nii.gz')]

postp = SegmentationPostprocessing(
                 apply_small_component_removing=True,
                 volume_thresholds=10,
                 remove_2d_comps=True,
                 use_fill_holes_2d=True,
                 lb_classes=[1,9])

for nii_file in tqdm(nii_files):
    
    img = nib.load(join(predp, nii_file))
    lb = img.get_fdata().astype(int)
    
    lb_clean = postp.remove_small_components(postp.fill_holes(lb, False))
    
    z_change = np.where(np.sum(np.abs(lb - lb_clean), (1,2)))[0]
    
    if len(z_change) == 0:
        continue
    
    scan = nii_file.split('.')[0]
    
    if scan.startswith('BARTS'):
        scan = scan[10:]
    else:
        scan = '_'.join(scan.split('_')[:-1])
    
    dcmp = join(rawp, scan)
    data_tpl = read_dcms(dcmp)
    im = data_tpl['image'].clip(-150, 250)
    
    for z in z_change:
        plt.subplot(1, 2, 1)
        plt.imshow(im[z], cmap='bone')
        plt.contour(lb[z] > 0)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(im[z], cmap='bone')
        plt.contour(lb_clean[z] > 0)
        plt.axis('off')
        plt.savefig(join(predp, 'changes', scan+'_'+str(z)), bbox_inches='tight')
        plt.close()
    
    nii_img = nib.Nifti1Image(lb_clean, img.affine, img.header)
    nib.save(nii_img, join(predp, nii_file))
    
    data_tpl['lb_clean'] = reduce_classes(lb_clean.astype(int), [1,9])
    
    out_file = join(predp, basename(data_tpl['raw_label_file']))
    
    save_dcmrt_from_data_tpl(data_tpl,
                             out_file=out_file,
                             key='lb_clean',
                             names=['1', '9'])
