import numpy as np
import torch


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
            elif torch.is_tensor(item1):
                if not np.all(item1.detach().cpu().numpy() == item2.detach().cpu().numpy()):
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


def print_dict_diff(dict1, dict2, dict1_name='input dict', dict2_name='loaded dict', pref=''):

    # check if keys are missing
    common_keys = []
    for key in dict1:
        if key not in dict2:
            print(pref+ key+' missing in '+dict2_name)
        else:
            common_keys.append(key)
    
    for key in dict2:
        if key not in dict1:
            print(pref+key+' missing in '+dict1_name)

    # type checking
    remaining_keys = []
    for key in common_keys:
        if type(dict1[key]) != type(dict2[key]):
            print(pref+key+' type missmatch, got {} for {} and {} for {}'.format(type(dict1[key]),
                                                                                 dict1_name,
                                                                                 type(dict2[key]),
                                                                                 dict2_name))
            print('Values: {}, {}'.format(dict1[key], dict2[key]))
        else:
            remaining_keys.append(key)

    # now we can check the content
    for key in remaining_keys:
        item1, item2 = dict1[key], dict2[key]
        try:
            if isinstance(item1, np.ndarray):
                if not np.all(item1 == item2):
                    print(pref+key+' missmatch: {}, {}'.format(item1, item2))
            elif torch.is_tensor(item1):
                if not np.all(item1.detach().cpu().numpy() == item2.detach().cpu().numpy()):
                    print(pref+key+' missmatch: {}, {}'.format(item1, item2))
            elif isinstance(item1, (list, tuple)):
                if not np.all(np.array(item1) == np.array(item2)):
                    print(pref+key+' missmatch: {}, {}'.format(item1, item2))
    
            elif isinstance(item1, dict):
                print_dict_diff(item1, item2, dict1_name, dict2_name, pref+key+' -> ')
            else:
                    if not item1 == item2:
                        print(pref+key+' missmatch: {}, {}'.format(item1, item2))
        except ValueError:
            print('Value Error when compating {} and {}.'.format(item1, item2))
            
    