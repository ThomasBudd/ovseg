import numpy as np
import torch
from ovseg.utils.torch_np_utils import stack, check_type

# just a collection of functions that create and manipulate 2d and 3d grids
# current implementation handels resizing, scaling and rotations


def get_centred_np_grid(shape, spacing=None, shape_new=None):
    if spacing is None:
        spacing = np.ones_like(shape)
    if shape_new is None:
        shape_new = shape
    axes = [np.linspace(-0.5*sp*(s-1), 0.5*sp*(s-1), int(sn))
            for s, sp, sn in zip(shape, spacing, shape_new)]
    return np.stack(np.meshgrid(*axes, indexing='ij'))


def get_centred_torch_grid(shape, spacing=None, shape_new=None, device='cpu'):
    if spacing is None:
        spacing = np.ones_like(shape)
    if shape_new is None:
        shape_new = shape
    axes = [torch.linspace(-0.5*sp*(s-1), 0.5*sp*(s-1), int(sn), device=device)
            for s, sp, sn in zip(shape, spacing, shape_new)]
    return torch.stack(torch.meshgrid(axes))


def get_resize_np_grid(shape_old, shape_new):
    axes = [np.linspace(0, so-1, int(sn))
            for so, sn in zip(shape_old, shape_new)]
    return np.stack(np.meshgrid(*axes, indexing='ij'))


def get_resize_torch_grid(shape_old, shape_new):
    axes = [torch.linspace(0, so-1, int(sn))
            for so, sn in zip(shape_old, shape_new)]
    return torch.stack(torch.meshgrid(axes))


def scale_grid(grid, scale, is_inverse=False):
    _, is_torch = check_type(grid)
    if is_torch:
        scale = torch.tensor(scale, device=grid.device)
    if is_inverse:
        return grid/scale
    else:
        return grid*scale


def rotate_axes(ax1, ax2, alpha):
    is_np, _ = check_type(ax1)
    if is_np:
        ax1_rot = np.cos(alpha) * ax1 + np.sin(alpha) * ax2
        ax2_rot = -1*np.sin(alpha) * ax1 + np.cos(alpha) * ax2
    else:
        alpha = torch.tensor(alpha, device=ax1.device)
        ax1_rot = torch.cos(alpha) * ax1 + torch.sin(alpha) * ax2
        ax2_rot = -1*torch.sin(alpha) * ax1 + torch.cos(alpha) * ax2

    return ax1_rot, ax2_rot


def rotate_grid_2d(grid, alpha, is_inverse=False):
    if is_inverse:
        return stack(rotate_axes(grid[0], grid[1], -1*alpha))
    else:
        return stack(rotate_axes(grid[0], grid[1], alpha))


def rotate_grid_3d(grid, alpha, is_inverse=False):
    if not len(alpha) == 3:
        raise ValueError('Input alphas must be of length 3 for 3d rotataions')
    gx, gy, gz = grid
    if is_inverse:
        gy, gz = rotate_axes(gy, gz, -1*alpha[0])
        gx, gz = rotate_axes(gx, gz, -1*alpha[1])
        gx, gy = rotate_axes(gx, gy, -1*alpha[2])
    else:
        gx, gy = rotate_axes(gx, gy, alpha[2])
        gx, gz = rotate_axes(gx, gz, alpha[1])
        gy, gz = rotate_axes(gy, gz, alpha[0])
    return stack([gx, gy, gz])


def grid_to_indices(grid, shape, spacing=None):
    if spacing is None:
        spacing = np.ones(len(grid))
    for d in range(len(grid)):
        grid[d] = (grid[d] + 0.5 * spacing[d] * (shape[d] - 1))/spacing[d]
    return grid


def get_rotated_scaled_np_grid(shape, scale, alpha, is_inverse=False):
    grid = get_centred_np_grid(shape)
    grid = scale_grid(grid, scale, is_inverse=is_inverse)
    if len(shape) == 2:
        grid = rotate_grid_2d(grid, alpha, is_inverse=is_inverse)
    elif len(shape) == 3:
        grid = rotate_grid_3d(grid, alpha, is_inverse=is_inverse)
    grid = grid_to_indices(grid, shape)
    return grid


def get_rotated_scaled_torch_grid(shape, scale, alpha, is_inverse=False):
    grid = get_centred_torch_grid(shape)
    grid = scale_grid(grid, scale, is_inverse=is_inverse)
    if len(shape) == 2:
        grid = rotate_grid_2d(grid, alpha, is_inverse=is_inverse)
    elif len(shape) == 3:
        grid = rotate_grid_3d(grid, alpha, is_inverse=is_inverse)
    grid = grid_to_indices(grid, shape)
    return grid
