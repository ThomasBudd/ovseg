import torch
import numpy as np


def stack(items, axis=0):
    # wrapps np.stack and torch.stack
    if isinstance(items[0], np.ndarray):
        # numpy interpolation
        return np.stack(items, axis=axis)
    elif torch.is_tensor(items[0]):
        # torch interpolation
        return torch.stack(items, dim=axis)
    else:
        # error
        raise ValueError('Input of stack must be np.ndarray or torch.tensor.')


def check_type(inpt):
    is_np = isinstance(inpt, np.ndarray)
    is_torch = torch.is_tensor(inpt)
    if not is_np and not is_torch:
        raise TypeError('Expected input to be np.ndarray or torch.tensor. '
                        'Got {}'.format(type(inpt)))
    return is_np, is_torch


def maybe_add_channel_dim(inpt):

    is_np, _ = check_type(inpt)
    if len(inpt.shape) == 3:
        if is_np:
            return inpt[np.newaxis]
        else:
            return inpt.unsqueeze(0)
    elif len(inpt.shape) == 4:
        return inpt
    else:
        raise ValueError('Expected input to be 3d or 4d, got {}d'.format(len(inpt.shape)))
