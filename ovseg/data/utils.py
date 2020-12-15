import numpy as np


def crop_and_pad_image(volume, coord, patch_size, padded_patch_size, mode='minimum'):
    '''
    crop_and_pad_image(volume, coord, patch_size, padded_patch_size)
    crops from a volume with
    Parameters
    ----------
    volume : 4d tensor
    coord : len 3
        upper left coordinate of the patch
    patch_size : len 3
        size of the patch before padding
    patch_size : len 3
        size of the padded patch
    Returns
    -------
    None.

    '''
    shape = np.array(volume.shape)

    # global coordinates, possible outside volume
    cmn_in = coord - (padded_patch_size - patch_size)//2
    cmx_in = cmn_in + padded_patch_size

    # clip the coordinates to not violate the arrays axes
    cmn_vol = np.maximum(cmn_in, 0)
    cmx_vol = np.minimum(cmx_in, shape)

    # let's cut out of the volume as much as we can
    crop = volume[cmn_vol[0]:cmx_vol[0], cmn_vol[1]:cmx_vol[1],
                  cmn_vol[2]:cmx_vol[2]]

    # now the padding
    pad_low = -1 * np.minimum(0, cmn_in)
    pad_up = np.maximum(0, cmn_in - cmx_vol)
    pad_width = [(pl, pu) for pl, pu in zip(pad_low, pad_up)]

    return np.pad(crop, pad_width, mode=mode)


def folds_to_splits(folds):
    splits = []
    for i in range(len(folds)):
        train = []
        for j in range(len(folds)):
            if i == j:
                val = folds[j]
            else:
                train.extend(folds[j])
        splits.append({'train': train, 'val': val})
    return splits


def split_scans_random_uniform(scans, n_folds=5, fixed_shuffle=True):

    if fixed_shuffle:
        # fix the splitting of the data
        scans = sorted(scans)
        np.random.seed(12345)
    np.random.shuffle(scans)
    # number of items in all but the last fold
    size_fold = int(len(scans) / n_folds)
    # folds 0,1,...,n_folds-2
    folds = [scans[i * size_fold: (i + 1) * size_fold] for i in
             range(n_folds - 1)]
    # fold n_fold -1
    folds.append(scans[(n_folds - 1) * size_fold:])
    return folds_to_splits(folds)


def split_scans_by_patient_id(scans, decode, n_folds=5, fixed_shuffle=True):
    raise NotImplementedError('This method is not implemented yet!')
