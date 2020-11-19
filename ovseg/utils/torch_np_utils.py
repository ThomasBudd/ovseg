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
        raise TypeError('Expected input to be np.ndarray or torch.tensor')
    return is_np, is_torch
