import numpy as np
from skimage.measure import label
from os import environ, listdir
from os.path import join
import nibabel as nib
from tqdm import tqdm

predp_pod = join(environ['OV_DATA_BASE'], 'predictions', 'OV04', 'pod_067',
                 'larger_res_encoder', 'cross_validation')
predp_om = join(environ['OV_DATA_BASE'], 'predictions', 'OV04', 'om_08',
                'res_encoder_no_prg_lrn', 'cross_validation')

pod_cases = listdir(predp_pod)
om_cases = listdir(predp_om)

def remove_small_lesions(seg, fac):
    comps = label(seg)
    for c in range(1, comps.max()+1):
        comp = comps == c
        if np.sum(comp) * fac < 1000:
            seg[comp] = 0
    return seg

# %%
n_scans, ovlp, ovlp_tr = 0, 0, 0

for i in tqdm(range(276)):
    case = 'case_%03d.nii.gz' % i

    if case in pod_cases and case in om_cases:
        n_scans += 1
        pod = nib.load(join(predp_pod, case)).get_fdata()
        om = nib.load(join(predp_om, case)).get_fdata()
        ovlp += np.max(pod * om)

print(ovlp / n_scans)


# %%
n_scans, ovlp, ovlp_tr = 0, 0, 0

for i in tqdm(range(276)):
    case = 'case_%03d.nii.gz' % i

    if case in pod_cases and case in om_cases:
        n_scans += 1
        img = nib.load(join(predp_pod, case))
        fac = np.prod(img.header['pixdim'][1:4])
        pod = remove_small_lesions(img.get_fdata(), fac)
        om = remove_small_lesions(nib.load(join(predp_om, case)).get_fdata(), fac)
        ovlp += np.max(pod * om)

print(ovlp / n_scans)
