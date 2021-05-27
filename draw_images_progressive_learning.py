from os import environ, listdir
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from ovseg.model.SegmentationModel import SegmentationModel
import torch
plt.close('all')
lw = 0.2

prepp = join(environ['OV_DATA_BASE'], 'preprocessed', 'OV04', 'pod_half')
plotp = 'D:\\PhD\\ICM\\Segmentation\\eval_effUNet\\example_images'
# plotp = 'D:\\PhD\\ICM\\Latex\\Presentation 2021 05 28 CIA talk'
case = 'case_015.npy'

im_full = np.load(join(prepp, 'images', case))
lb_full = np.load(join(prepp, 'labels', case))

z = np.argmax(np.sum(lb_full, (1, 2)))

# %%
x, y = 24, 24
im = im_full[z, x:x+192, y:y+192].astype(float)
lb = lb_full[z, x:x+192, y:y+192].astype(float)

plt.imshow(im, cmap='gray')
plt.contour(lb, colors='red', linwidths=0.5)
plt.plot([16, 16, 176, 176, 16], [16, 176, 176, 16, 16], 'b')
plt.plot([32, 32, 160, 160, 32], [32, 160, 160, 32, 32], 'g')
plt.savefig(join(plotp, 'patch_sizes.png'), bbox_inches='tight')

# %%
x, y = 56, 56

im = im_full[z-16:z+16, x:x+128, y:y+128].astype(float)
lb = lb_full[z-16:z+16, x:x+128, y:y+128].astype(float)

sizes = [[24, 96, 96], [16, 64, 64]]
# %%
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(im[16], cmap='gray')
plt.contour(lb[16], colors='red', linwidths=lw)
plt.title('{}x{}x{}'.format(128, 128, 32))

for i, size in enumerate(sizes):
    im_rsz = resize(im, size, order=1)
    lb_rsz = resize(lb, size, order=0) > 0.5
    plt.subplot(1, 3, 2 + i)
    plt.imshow(im_rsz[size[0] // 2], cmap='gray')
    plt.contour(lb_rsz[size[0] // 2], colors='red', linwidths=lw)
    plt.title('{}x{}x{}'.format(size[1], size[2], size[0]))

plt.savefig(join(plotp, 'progressize_resizing.png'), bbox_inches='tight')

# %%
m = 10
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(im[16] + m/30 * np.random.randn(128, 128), cmap='gray')
plt.contour(lb[16], colors='red', linwidths=lw)
plt.title('{}x{}x{}'.format(128, 128, 32))

for i, size in enumerate(sizes):
    im_rsz = resize(im, size, order=1)
    lb_rsz = resize(lb, size, order=0) > 0.5
    plt.subplot(1, 3, 2 + i)
    plt.imshow(im_rsz[size[0] // 2] + m/30 * np.random.randn(*size[1:]), cmap='gray')
    plt.contour(lb_rsz[size[0] // 2], colors='red', linwidths=lw)
    plt.title('{}x{}x{}'.format(size[1], size[2], size[0]))

plt.savefig(join(plotp, 'progressize_resizing_aug.png'), bbox_inches='tight')
# %%
m = 10
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(im[16] + m/30 * np.random.randn(128, 128), cmap='gray')
plt.contour(lb[16], colors='red', linwidths=0.5)
plt.title('{}x{}x{}'.format(128, 128, 32))

for i, size in enumerate(sizes):
    im_rsz = resize(im, size, order=1)
    lb_rsz = resize(lb, size, order=0) > 0.5
    plt.subplot(1, 3, 2 + i)
    plt.imshow(im_rsz[size[0] // 2] + m/2**(i+1)/30 * np.random.randn(*size[1:]), cmap='gray')
    plt.contour(lb_rsz[size[0] // 2], colors='red', linwidths=lw)
    plt.title('{}x{}x{}'.format(size[1], size[2], size[0]))

plt.savefig(join(plotp, 'progressize_learning.png'), bbox_inches='tight')

# %% plot network outcome
model = SegmentationModel(5, model_name='lr_schedule_0.02', preprocessed_name='pod_half',
                          data_name='OV04', is_inference_only=True)
batch = torch.from_numpy(im_full[np.newaxis, np.newaxis,z-16:z+16, x:x+128, y:y+128]).type(torch.float).cuda()
with torch.no_grad():
    pred = model.network(batch)

# %%
for i in range(4):
    plt.figure()
    p = torch.nn.functional.softmax(pred[i], 1)
    p = p.cpu().numpy()[0, 1]
    p = p[p.shape[0] // 2]
    plt.imshow(p, cmap='gray')
    plt.axis('off')
    plt.savefig(join(plotp, 'pred_scale_{}.png'.format(i)), bbox_inches='tight')


im = im_full[z, x:x+128, y:y+128].astype(float)
lb = lb_full[z, x:x+128, y:y+128].astype(float)
plt.figure()
plt.subplot(2, 1, 1)
plt.imshow(im, cmap='gray')
plt.contour(lb, colors='red', linwidths=lw)
plt.axis('off')
plt.subplot(2, 1, 2)
plt.imshow(lb, cmap='gray')
plt.axis('off')
plt.savefig(join(plotp, 'target.png'), bbox_inches='tight')

plt.figure()
plt.imshow(im, cmap='gray')
plt.axis('off')
plt.savefig(join(plotp, 'input.png'), bbox_inches='tight')