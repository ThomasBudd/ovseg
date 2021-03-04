from os import listdir
from os.path import join, isdir
import pydicom
import nibabel as nib
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json

bp = 'D:\\PhD\\Data\\Barts_DICOM_Segmentation'
bnp = 'D:\\PhD\\Data\\NEW_Barts_segmentations_VB_RW'

scans = [s for s in listdir(bnp) if s.startswith('ID')]

observers = []

for scan in scans:
    subdirs = [d for d in listdir(join(bp, scan)) if isdir(join(bp, scan, d))]
    if len(subdirs) > 0:
        dcmp = join(bp, scan, subdirs[0])
    else:
        dcmp = join(bp, scan)

    cdcms = [dcm for dcm in listdir(dcmp) if dcm.startswith('ID')]
    if len(cdcms) == 1:
        cds = pydicom.dcmread(join(dcmp, cdcms[0]))
    observers.append(str(cds.RTROIObservationsSequence[0].ROIInterpreter))

# %%
roxana_scans = [scan for scan, observer in zip(scans, observers)
                if observer == 'Pintican^Roxana^^Dr']
print('Roxana baseline scans: {}'.format(len([s for s in roxana_scans if s.endswith('_1')])))
print('Roxana follow-up scans: {}'.format(len([s for s in roxana_scans if s.endswith('_2')])))

decode = json.load(open('D:\\PhD\\Data\\ovarian_nifti\\decode.json', 'rb'))
roxana_bl_ids = [s[3:-2] for s in roxana_scans if s.endswith('_1')]
roxana_bl_cases = []
for key in decode:
    info = decode[key]
    if info['dataset'] == 'BARTS' and info['timepoint'] == 'BL' and info['pat_id'] in roxana_bl_ids:
        roxana_bl_cases.append('case_'+key+'.nii.gz')

# %%
nii_new_path = 'D:\\PhD\\Data\\ov_data_base\\raw_data\\BARTS\\labels'
nii_old_path = 'D:\\PhD\\Data\\ov_data_base\\raw_data\\BARTS_old\\labels'

all_cases = listdir(nii_new_path)

for cases in [roxana_bl_cases, all_cases]:
    dices_pod = []
    dices_om = []
    op, vp1, vp2 = 0, 0, 0
    oo, vo1, vo2 = 0, 0, 0
    for case in tqdm(cases):
        lb1 = nib.load(join(nii_new_path, case)).get_fdata()
        lb2 = nib.load(join(nii_old_path, case)).get_fdata()
        pod1, pod2 = (lb1 == 9).astype(float), (lb2 == 9).astype(float)
        om1, om2 = (lb1 == 1).astype(float), (lb2 == 1).astype(float)

        if pod1.max() > 0 or pod2.max() > 0:
            dices_pod.append(200 * np.sum(pod1 * pod2) / np.sum(pod1 + pod2))
        if om1.max() > 0 or om2.max() > 0:
            dices_om.append(200 * np.sum(om1 * om2) / np.sum(om1 + om2))
        op += np.sum(pod1 * pod2)
        vp1 += np.sum(pod1)
        vp2 += np.sum(pod2)
        oo += np.sum(om1 * om2)
        vo1 += np.sum(om1)
        vo2 += np.sum(om2)

    DSC_pod = 200 * op / (vp1 + vp2)
    DSC_om = 200 * oo / (vo1 + vo2)

    print('Dices POD: {:.3f} +- {:.3f}'.format(np.mean(dices_pod), np.std(dices_pod)))
    print('Dices Omentum: {:.3f} +- {:.3f}'.format(np.mean(dices_om), np.std(dices_om)))

    print('Globale DSC: POD: {:.3f}, omentum: {:.3f}'.format(DSC_pod, DSC_om))

# %%
plt.subplot(121)
plt.hist(dices_pod)
plt.title('POD')
plt.xlabel('DSC')
plt.ylabel('frequency')
plt.subplot(122)
plt.hist(dices_om)
plt.title('Omentum')
plt.xlabel('DSC')
plt.ylabel('frequency')

# %%
predp = 'E:\\PhD\\Data\\nnUNet_raw_data_base\\nnUNet_predictions'
dices_pod = []
dices_om = []
op, vp1, vp2 = 0, 0, 0
oo, vo1, vo2 = 0, 0, 0
for case in tqdm(roxana_bl_cases):
    lb1 = nib.load(join(nii_new_path, case)).get_fdata()
    pod1, om1 = (lb1 == 9).astype(float), (lb1 == 1).astype(float)
    pod2 = nib.load(join(predp, '140', case)).get_fdata()
    om2 = nib.load(join(predp, '141', case)).get_fdata()

    if pod1.max() > 0 or pod2.max() > 0:
        dices_pod.append(200 * np.sum(pod1 * pod2) / np.sum(pod1 + pod2))
    if om1.max() > 0 or om2.max() > 0:
        dices_om.append(200 * np.sum(om1 * om2) / np.sum(om1 + om2))
    op += np.sum(pod1 * pod2)
    vp1 += np.sum(pod1)
    vp2 += np.sum(pod2)
    oo += np.sum(om1 * om2)
    vo1 += np.sum(om1)
    vo2 += np.sum(om2)

DSC_pod = 200 * op / (vp1 + vp2)
DSC_om = 200 * oo / (vo1 + vo2)

print('Dices POD: {:.3f} +- {:.3f}'.format(np.mean(dices_pod), np.std(dices_pod)))
print('Dices Omentum: {:.3f} +- {:.3f}'.format(np.mean(dices_om), np.std(dices_om)))

print('Globale DSC: POD: {:.3f}, omentum: {:.3f}'.format(DSC_pod, DSC_om))
