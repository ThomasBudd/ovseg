import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

patch_size = [56, 196, 160]
sigma_scale = 1/8
tmp = np.zeros(patch_size)
center_coords = [i // 2 for i in patch_size]
sigmas = [i * sigma_scale for i in patch_size]
tmp[tuple(center_coords)] = 1
gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
gaussian_importance_map = gaussian_importance_map.astype(np.float32)

# gaussian_importance_map cannot be 0, otherwise we may end up with nans!
gaussian_importance_map[gaussian_importance_map == 0] = np.min(
    gaussian_importance_map[gaussian_importance_map != 0])

plt.imshow(gaussian_importance_map[56//2], cmap='gray')


# %%
patch_size = np.array([56, 192, 160])
shape = np.array([64, 256, 256])
nz, nx, ny = shape
overlap = 0.5
# upper left corners of all patches
z_list = list(range(0, nz - patch_size[0], max([int(patch_size[0] * overlap), 1]))) \
    + [nz - patch_size[0]]
x_list = list(range(0, nx - patch_size[1], max([int(patch_size[1] * overlap), 1]))) \
    + [nx - patch_size[1]]
y_list = list(range(0, ny - patch_size[2], max([int(patch_size[2] * overlap), 1]))) \
    + [ny - patch_size[2]]

#
n_patches = np.ceil((shape - patch_size) / (overlap * patch_size)).astype(int) + 1

image_size = shape
step_size = overlap
target_step_sizes_in_voxels = [i * step_size for i in patch_size]

num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

steps = []
for dim in range(len(patch_size)):
    # the highest step value for this dimension is
    max_step_value = image_size[dim] - patch_size[dim]
    if num_steps[dim] > 1:
        actual_step_size = max_step_value / (num_steps[dim] - 1)
    else:
        actual_step_size = 99999999999  # does not matter because there is only one step at 0

    steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

    steps.append(steps_here)