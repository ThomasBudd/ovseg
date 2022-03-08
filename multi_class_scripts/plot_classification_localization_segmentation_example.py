import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
plt.close()
from skimage.measure import label
import os
p = 'D:\\PhD\\kits19\\data\\case_00000'

plotp = os.path.join(os.environ['OV_DATA_BASE'], 'plots')

fs = 18

im = nib.load(p + '\\imaging.nii.gz').get_fdata().clip(-150, 250)
im = (im + 150) / 400
seg = nib.load(p + '\\segmentation.nii.gz').get_fdata()

z1 = np.argmax(np.sum(seg == 2, (1,2)))
z1 = 295
im1 = im[z1, 150:470,  100:420]
seg1 = seg[z1, 150:470,  100:420]

z2 = 200
im2 = im[z2, 150:470,  100:420]
seg2 = seg[z2, 150:470,  100:420]

def plot_overlay(im, seg1, seg2=None, v=0.3):
    
    seg1 = seg1.astype(float)
    im = (im - im.min()) / (im.max() - im.min())
    if seg2 is None:
        overlay = np.stack([im + v * seg1, im, im], -1) /(1+v)
    else:
        seg2 = seg2.astype(float)
        overlay= np.stack([im + v * seg1, im, im + v*seg2], -1) /(1+v)
    plt.imshow(overlay)

def plot_bboxes(seg, color='r'):
    ccomps = label(seg)
    n_comps = ccomps.max()
    for c in range(1, n_comps+1):
        comp = (ccomps == c)
        x, y = np.where(comp)
        ymn, ymx, xmn, xmx = x.min(), x.max(), y.min(), y.max()
        plt.plot([xmn, xmn], [ymn, ymx], color)
        plt.plot([xmn, xmx], [ymx, ymx], color)
        plt.plot([xmx, xmx], [ymx, ymn], color)
        plt.plot([xmx, xmn], [ymn, ymn], color)

# %%
# plt.subplot(1, 4, 1)
# plot_overlay(im1, seg1==1)
plt.imshow(im1, cmap='gray', vmax=1.3)
# plt.title('Classification', fontsize=fs)
# plt.xlabel('Kidney', fontsize=fs)
plt.yticks([])
plt.xticks([])
plt.savefig(os.path.join(plotp, 'Classification.png'), pad_inches=0)
plt.close()
# plt.subplot(2, 3, 4)
# # plot_overlay(im2, seg2==1)
# plt.imshow(im2, cmap='gray')
# plt.xlabel('No Kidney', fontsize=fs)
# plt.yticks([])
# plt.xticks([])
# plt.subplot(1, 4, 2)
# plt.imshow(im1, cmap='gray', vmax=1.3)
# plt.title('Localization', fontsize=fs)
# plot_bboxes(seg1 == 1)
# plt.axis('off')
# plt.subplot(1, 4, 3)
plt.imshow(im1, cmap='gray', vmax=1.3)
# plt.title('Localization', fontsize=fs)
plot_bboxes(seg1 == 1)
plot_bboxes(seg1 == 2, 'b')
plt.axis('off')
plt.savefig(os.path.join(plotp, 'Detection.png'), pad_inches=0)
plt.close()
# plt.subplot(2, 3, 5)
# plt.imshow(im2, cmap='gray')
# plt.axis('off')
# plt.subplot(1, 4, 4)
plot_overlay(im1, seg1 == 1, seg1 == 2)
plt.axis('off')
plt.savefig(os.path.join(plotp, 'Segmentation.png'), pad_inches=0)
plt.close()
