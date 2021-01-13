import numpy as np


def dict_equal(dict1, dict2):

    if not dict1.keys() == dict2.keys():
        return False

    keys = dict1.keys()

    is_equal = [dict1[key] == dict2[key] for key in keys]
    for i, key in enumerate(keys):
        if isinstance(is_equal[i], np.ndarray):
            is_equal[i] = np.all(is_equal[i])
    return np.all(is_equal)
