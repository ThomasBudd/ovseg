import numpy as np


def dict_equal(dict1, dict2):

    if not dict1.keys() == dict2.keys():
        return False

    keys = dict1.keys()

    for key in keys:
        item1, item2 = dict1[key], dict2[key]
        if not isinstance(item1, type(item2)):
            return False

        try:
            if isinstance(item1, np.ndarray):
                if not np.all(item1 == item2):
                    return False
            elif isinstance(item1, (list, tuple)):
                if not np.all(np.array(item1) == np.array(item2)):
                    return False
    
            elif isinstance(item1, dict):
                if not dict_equal(item1, item2):
                    return False
            else:
                    if not item1 == item2:
                        return False
        except ValueError:
            print('Value Error when compating {} and {}.'.format(item1, item2))
            return False

    return True
