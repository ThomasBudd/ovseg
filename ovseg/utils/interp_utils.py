import torch
import torch.nn.functional as F
import numpy as np
try:
    from scipy.ndimage import map_coordinates
except ImportError:
    print('Caught Import Error while importing some function from scipy or skimage (scikit-image). '
          'Please use a newer version of gcc.')
from ovseg.utils.grid_utils import get_resize_np_grid, get_resize_torch_grid
from ovseg.utils.torch_np_utils import stack, check_type

# conventions and definitions:
# a batch is a collection of samples
# a sample is a collection of images
# an image has the dimensions (nx, ny) or (nx, ny, nz)
# this script holds functions for batch, sample and image interpolation.
# it is assumed that for each sample in the batch a grid is given
# and that each image in a sample is transformed with the same grid

print_torch_order_warning = True


def torch_interp_img(img, grid, order, cval=None):
    '''
    Performs 2d, 2.5d and 3d nearest neighbour or (bi/tri)-linear
    interpolation for a torch tensor.
    For 2.5d interpolation it is assumed that the last axes is the
    z axes.

    Parameters
    ----------
    img : torch.tensor
        2d or 3d image [nx, ny(, nz)].
    grid : torch.tensor
        grid of shape (2,nx,ny) or (3,nx,ny,z) for interpolation.
    odrer : 0,1
        0 for nearest neighbour and 1 for linear interpolation
    cval : scalar
        padding value for the boundary, default minimum of img

    Returns
    -------
    img : torch.tensor
        image in new coordinates.

    '''
    global print_torch_order_warning
    if order > 1:
        order = 1
        if print_torch_order_warning:
            print('WARNING: torch interpolation was called with order>1.\n'
                  'The order will be reduced to 1 in this and future calles')
            print_torch_order_warning = False

    if not torch.is_tensor(img):
        raise ValueError('Input img must be torch tensor.')

    if not torch.is_tensor(grid):
        raise ValueError('Input grid must be torch tensor.')

    shape = np.array(img.shape)
    dim = len(shape)
    if not len(shape) in [2, 3]:
        raise ValueError('Input img is expected to be 2d or 3d.')

    if not grid.shape[0] == len(grid.shape)-1:
        raise ValueError('grid must be of shape (2,nx,ny) for 2d or '
                         + '(3,nx,ny,nz) for 3d interpolation.'
                         ' Got {}'.format(grid.shape))
    # interpolation dimension
    idim = grid.shape[0]

    # if no cval is giving we use the smallest value for padding
    if cval is None:
        cval = img.min().item()

    # pad the image with extrapolation values
    pad = [0 for _ in range(2*dim-2*idim)] + [1 for _ in range(2*idim)]
    img_pad = F.pad(img, pad, value=cval)
    # padding shifts the grid values by one
    grid = grid + 1

    if order == 0:
        # do nearest neighbour interpolation
        # clamp the grid values to make sure we're using values outside
        # the index range
        grid = torch.stack([torch.clamp(grid[i], 0, shape[i]+1)
                            for i in range(idim)])
        inds = torch.round(grid).long()
        if idim == 2:
            img_trsf = img_pad[inds[0], inds[1]]
        elif idim == 3:
            img_trsf = img_pad[inds[0], inds[1], inds[2]]
        else:
            raise ValueError('grid must be of shape (2,nx,ny) for 2d or '
                             + '(3,nx,ny,nz) for 3d interpolation.'
                             ' Got {}'.format(grid.shape))

    elif order == 1:
        # do bi or triliner interpolation
        img_pad = img_pad.type(img.dtype)
        # cample grid (see above)
        grid = torch.stack([torch.clamp(grid[i], 0, shape[i]) for i in range(idim)])
        inds = torch.floor(grid).long()
        xi = (grid - inds).type(img.dtype)
        if idim == 2:

            if dim == 3:
                xi = xi.unsqueeze(-1)

            img_trsf = (1 - xi[0]) * (1 - xi[1]) * img_pad[inds[0], inds[1]]
            img_trsf = img_trsf + (1 - xi[0]) * xi[1] * img_pad[inds[0], inds[1] + 1]
            img_trsf = img_trsf + xi[0] * (1 - xi[1]) * img_pad[inds[0] + 1, inds[1]]
            img_trsf = img_trsf + xi[0] * xi[1] * img_pad[inds[0] + 1, inds[1] + 1]

        elif idim == 3:

            img_trsf = (1 - xi[0]) * (1 - xi[1]) * (1 - xi[2]) * img_pad[inds[0], inds[1], inds[2]]
            img_trsf = img_trsf + xi[0] * (1 - xi[1]) * (1 - xi[2]) * img_pad[inds[0] + 1, inds[1], inds[2]]
            img_trsf = img_trsf + (1 - xi[0]) * xi[1] * (1 - xi[2]) * img_pad[inds[0], inds[1] + 1, inds[2]]
            img_trsf = img_trsf + xi[0] * xi[1] * (1 - xi[2]) * img_pad[inds[0] + 1, inds[1] + 1, inds[2]]
            img_trsf = img_trsf + (1 - xi[0]) * (1 - xi[1]) * xi[2] * img_pad[inds[0], inds[1], inds[2]+1]
            img_trsf = img_trsf + xi[0] * (1 - xi[1]) * xi[2] * img_pad[inds[0] + 1, inds[1], inds[2]+1]
            img_trsf = img_trsf + (1 - xi[0]) * xi[1] * xi[2] * img_pad[inds[0], inds[1] + 1, inds[2]+1]
            img_trsf = img_trsf + xi[0] * xi[1] * xi[2] * img_pad[inds[0] + 1, inds[1] + 1, inds[2]+1]

        else:
            raise ValueError('grid must be of shape (2,nx,ny) for 2d or '
                             + '(3,nx,ny,nz) for 3d interpolation.'
                             ' Got {}'.format(grid.shape))
    else:
        raise ValueError('torch_interp_img is only implemented for orders 0 '
                         'and 1')
    return img_trsf


def np_interp_img(img, grid, order, cval=None):
    '''
    Performs 2d, 2.5d and 3d spline interpolation for np arrays.
    For 2.5d interpolation it is assumed that the last axes is the
    z axes.

    Parameters
    ----------
    img : np.ndarray
        2d or 3d image [nx, ny(, nz)].
    grid : np.ndarray
        grid of shape (2,nx,ny) or (3,nx,ny,z) for interpolation.
    odrer : 0,1,3
        spline order of interpoltion
    cval : scalar
        padding value for the boundary, default minimum of img

    Returns
    -------
    img : np.ndarray
        image in new coordinates.

    '''
    if not isinstance(img, np.ndarray):
        raise ValueError('Input img must be numpy array.')

    if not isinstance(grid, np.ndarray):
        raise ValueError('Input grid must be numpy array.')

    img = img.astype(np.float32)
    shape = np.array(img.shape)
    dim = len(shape)
    if not len(shape) in [2, 3]:
        raise ValueError('Input img is expected to be 2d or 3d.')

    if not grid.shape[0] == len(grid.shape)-1:
        raise ValueError('grid must be of shape (2,nx,ny) for 2d or '
                         + '(3,nx,ny,nz) for 3d interpolation.')
    # interpolation dimension
    idim = grid.shape[0]

    # if no cval is giving we use the smallest value for padding
    if cval is None:
        cval = img.min()

    if dim == idim:
        # perform simple 2d or 3d interpolation via map_coordinates
        return map_coordinates(img, grid, order=order, cval=cval)

    elif dim == 3 and idim == 2:
        # 2.5d interpolation, we will perform interpolation onyl in xy plane
        return np.stack([map_coordinates(img[..., z], grid,
                                         order=order, cval=cval)
                         for z in range(img.shape[-1])], -1)


def interp_img(img, grid, order, cval=None):
    '''
    Wrapper for torch_interp_img and np_interp_img
    '''
    is_np, _ = check_type(img)
    if is_np:
        return np_interp_img(img, grid, order, cval)
    else:
        return torch_interp_img(img, grid, order, cval)


def resize_img(img, shape_new, order):
    '''
    Parameters
    ----------
    img : np array or torch tensor
        [nx, ny(, nz)]
    shape_new : list, tuple
        length 2 or 3
    orders : int
        spline order of interpoliation

    Returns
    -------
    None.

    '''
    shape_old = np.array(img.shape)
    is_np, _ = check_type(img)
    if is_np:
        grid = get_resize_np_grid(shape_old, shape_new)
        return np_interp_img(img, grid, order)
    else:
        grid = get_resize_torch_grid(shape_old, shape_new).to(img.device)
        return torch_interp_img(img, grid, order)


def change_img_pixel_spacing(img, spc_old, spc_new, order):
    '''

    Parameters
    ----------
    img : np array or torch tensor
        [nx, ny(, nz)]
    spc_old : list or tuple
        old pixel spacing.
    spc_new : list or tuple
        new pixel spacing
    order : int
        spline interpolation order

    Returns
    -------
    None.

    '''
    spc_old = np.array(spc_old)
    spc_new = np.array(spc_new)
    shape_new = np.round(np.array(img.shape)*spc_old/spc_new)
    return resize_img(img, shape_new, order)
# %% now function for sample interpolation


def interp_sample(sample, grid, orders, cvals=None):
    '''

    Parameters
    ----------
    sample : np array or torch tensor
        [channels, nx, ny(,nz)].
    grids : np array or torch tensor
        [dim, nx, ny(, nz)], dim=2,3
    orders : scalar, list or tuple
        interploation order for each channel/image
    cvals : scalar, list or tuple
        extrapolation values for each channel/image

    Returns
    -------
    np array or torch tensor
        batch of transformed samples

    '''
    ssize = sample.shape[0]
    if not isinstance(cvals, (list, tuple, np.ndarray)):
        cvals = [cvals for _ in range(ssize)]
    assert len(cvals) == sample.shape[0]
    if not isinstance(orders, (list, tuple, np.ndarray)):
        orders = [orders for _ in range(ssize)]
    assert len(cvals) == sample.shape[0]
    assert len(orders) == sample.shape[0]

    return stack([interp_img(sample[i], grid, orders[i], cval=cvals[i])
                  for i in range(ssize)])


def resize_sample(sample, shape_new, orders):
    '''
    Parameters
    ----------
    sample : np array or torch tensor
        [channels, nx, ny(, nz)]
    shape_new : list, tuple
        length 2 or 3
    orders : int or list
        spline order of interpoliation

    Returns
    -------
    None.

    '''
    shape_old = np.array(sample.shape[1:])
    is_np, _ = check_type(sample)
    if is_np:
        grid = get_resize_np_grid(shape_old, shape_new)
    else:
        grid = get_resize_torch_grid(shape_old, shape_new).to(sample.device)
    return interp_sample(sample, grid, orders)


def change_sample_pixel_spacing(sample, spc_old, spc_new, orders):
    '''

    Parameters
    ----------
    sample : np array or torch tensor
        [channels, nx, ny(, nz)]
    spc_old : list or tuple
        old pixel spacing.
    spc_new : list or tuple
        new pixel spacing
    orders : int or list/tuple
        spline interpolation order

    Returns
    -------
    None.

    '''
    spc_old = np.array(spc_old)
    spc_new = np.array(spc_new)
    shape_new = np.round(np.array(sample.shape[1:])*spc_old/spc_new)
    return resize_sample(sample, shape_new, orders)
# %% last but not least the function for batch interpolation


def interp_batch(batch, grids, orders, cvals=None):
    '''
    Parameters
    ----------
    batch : np array or torch tensor
        [batch_size, channels, nx, ny(,nz)].
    grids : np array or torch tensor
        [batch_size, dim, nx, ny(, nz)], dim=2,3
    orders : scalar, list or tuple
        interploation order for each channel/image
    cvals : scalar, list or tuple
        extrapolation values for each channel/image

    Returns
    -------
    np array or torch tensor
        batch of transformed samples

    '''
    assert batch.shape[0] == grids.shape[0]
    return stack([interp_sample(batch[i], grids[i], orders, cvals)
                  for i in range(batch.shape[0])])


def resize_batch(batch, shape_new, orders):
    '''
    resizes all samples in the batch to shape_new
    Parameters
    ----------
    batch : list or tuple of np arrays or torch tensors
        samples with possibly differnet shapes
    shape_new : shape
        new shape after resizing
    orders : scalar or list
        spline order used for interpolation of each image in a sample

    Raises
    ------
    ValueError
        if batch is not a list or tuple

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    if not isinstance(batch, (tuple, list)):
        raise ValueError('Input to resize batch must be batch items in a '
                         'list or tuple.')
    return stack([resize_sample(batch[i], shape_new, orders)
                  for i in range(len(batch))])


def change_batch_pixel_spacing(batch, spcs_old, spc_new, orders):
    '''
    resizes all samples in the batch from their old to a new spacing
    Parameters
    ----------
    batch : list, tuple,  np array or torch tensor
        samples with possibly differnet shapes
    spcs_old : list, tuple, np.ndarray
        old/current spacings
    orders : scalar or list
        spline order used for interpolation of each image in a sample

    Raises
    ------
    ValueError
        if batch is not a list or tuple
        if spcs old has not the same length as batch

    Returns
        list of samples in new spacing,
            it is not stacked to a array or tensor because the new shapes
            might differ
    -------
    TYPE
        DESCRIPTION.

    '''
    if not (isinstance(batch, (tuple, list, np.ndarray))
            or torch.is_tensor(batch)):
        raise ValueError('Input to resize batch must be batch items in a '
                         'list, tuple, np.ndarray or torch.tensor.')
    if not isinstance(spcs_old, (tuple, list, np.ndarray)):
        raise ValueError('Input to resize batch must be batch items in a '
                         'list or tuple.')
    if len(batch) != len(spcs_old):
        raise ValueError('batch and spcs_old must have same length')

    return [change_sample_pixel_spacing(b, s, spc_new, orders)
            for b, s in zip(batch, spcs_old)]
