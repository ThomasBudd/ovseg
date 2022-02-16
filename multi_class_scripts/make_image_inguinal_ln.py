import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os

rawp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data',
                    'OV04')

for scan in os.listdir(os.path.join(rawp, 'labels')):
    
    lb = nib.load(os.path.join(rawp, 'labels', scan)).get_fdata()
    seg = (lb == 17).astype(float)
    if seg.max() > 0:
        break

im = nib.load(os.path.join(rawp, 'images', scan.split('.')[0]+'_0000.nii.gz')).get_fdata()
im = (im.clip(-150, 250) + 150)/400
z = np.argmax(np.sum(seg, (0,1)))

plt.close()
plt.subplot(1, 2, 1)
plt.imshow(im[100:-100,100:-100,z], cmap='bone')
plt.contour(seg[100:-100,100:-100, z]>0, colors='red')
plt.axis('off')

for scan in os.listdir(os.path.join(rawp, 'labels'))[::-1]:
    
    lb = nib.load(os.path.join(rawp, 'labels', scan)).get_fdata()
    seg = (lb == 17).astype(float)
    if seg.max() > 0:
        break

im = nib.load(os.path.join(rawp, 'images', scan.split('.')[0]+'_0000.nii.gz')).get_fdata()
im = (im.clip(-150, 250) + 150)/400
z = np.argmax(np.sum(seg, (0,1)))
plt.subplot(1, 2, 2)
plt.imshow(im[100:-100,100:-100,z], cmap='bone')
plt.contour(seg[100:-100,100:-100, z]>0, colors='red')
plt.axis('off')