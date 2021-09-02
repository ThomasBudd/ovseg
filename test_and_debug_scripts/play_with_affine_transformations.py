import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

im_full = np.load('D:\\PhD\\Data\\ov_data_base\\preprocessed\\OV04_test\\default\\images'
                  '\\OV04_034_20091014.npy')
lb_full = np.load('D:\\PhD\\Data\\ov_data_base\\preprocessed\\OV04_test\\default\\labels'
                  '\\OV04_034_20091014.npy')

im_crop = im_full[20:68, 100:292, 100:292].astype(np.float32)
imt = torch.from_numpy(im_crop).cuda().unsqueeze(0).unsqueeze(0).type(torch.float)
lb_crop = lb_full[20:68, 100:292, 100:292].astype(np.float32)
lbt = torch.from_numpy(lb_crop).cuda().unsqueeze(0).unsqueeze(0).type(torch.float)
angle = 10

theta = torch.zeros((2, 3, 4))

angles = [-10, 10]
i1, i2, i3 = 0, 1, 2
for i, angle in enumerate(angles):
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    theta[i, i1, i1] = c
    theta[i, i1, i2] = s
    theta[i, i2, i1] = -1*s
    theta[i, i2, i2] = c
    theta[i, i3, i3] = 1

# for i in range(2):
#     for k in range(3):
#         theta[i, k, k] = 1
#         if i == 0:
#             theta[i, 1, 2] = 0.1
#         else:
#             theta[i, 2, 1] = 0.1
#         theta[i, i, 3] = 0


grid = F.affine_grid(theta, [2, 1, 48, 192, 192]).cuda()

im_trsf = F.grid_sample(torch.cat([imt, imt]), grid).cpu().numpy()
lb_trsf = F.grid_sample(torch.cat([lbt, lbt]), grid).cpu().numpy()

z = np.argmax(np.sum(lb_crop > 0, (1, 2)))
plt.subplot(1, 3, 1)
plt.imshow(im_trsf[0, 0, z], cmap='gray')
# plt.contour(lb_trsf[0, 0, z - 1 + i] > 0)
plt.subplot(1, 3, 2)
plt.imshow(im_crop[z], cmap='gray')
# plt.contour(lb_crop[z - 1 + i] > 0)
plt.subplot(1, 3, 3)
plt.imshow(im_trsf[1, 0, z], cmap='gray')
# plt.contour(lb_trsf[1, 0, z - 1 + i] > 0)


# %%
