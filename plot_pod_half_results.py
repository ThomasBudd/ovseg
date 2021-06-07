import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

raw_p = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'BARTS')
predp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions', 'OV04', 'pod_half',
                     'res_encoder_p_bias_0.5', 'BARTSensemble_0_1_2_3_4')
plotp = os.path.join(os.environ['OV_DATA_BASE'], 'plots', 'OV04', 'pod_half',
                     'res_encoder_p_bias_0.5')


for case in os.listdir(predp):
    name = case[:8]
    casefol = os.path.join(plotp, name)
    if not os.path.exists(casefol):
        os.makedirs(casefol)
    im = nib.load(os.path.join(raw_p, 'images', name+'_0000.nii.gz')).get_fdata()
    lb = (nib.load(os.path.join(raw_p, 'labels', case)).get_fdata() == 9).astype(int)
    pred = nib.load(os.path.join(predp, case)).get_fdata()

    contains = np.where(np.sum(lb+pred, (0,1)))[0]

    for z in contains:
        plt.imshow(im[..., z].clip(-150, 250), cmap='gray')
        plt.contour(lb[..., z], colors='red')
        plt.contour(pred[..., z], colors='blue')
        plt.savefig(os.path.join(casefol, str(z)), bbox_inches='tight')
        plt.close()