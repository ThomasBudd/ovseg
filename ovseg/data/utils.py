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


def split_scans_by_patient_id(scans, patient_ids, n_folds=4,
                              fixed_shuffle=True):

    # first we check if the patient ID dict is like we want it
    if not isinstance(patient_ids, dict):
        raise TypeError('patient_ids must be dict. The keys must be the names '
                        'of the files and the items the patient ids.')

    # let's see if all the scans are in the dict
    not_in_patient_ids = []
    for scan in scans:
        if scan not in patient_ids:
            not_in_patient_ids.append(scan)
    if len(not_in_patient_ids) > 0:
        raise ValueError('Some names of scans were not found in the '
                         'patient_ids: \n' + str(not_in_patient_ids))

    # shuffle the scans either randomly of fixed at random
    if fixed_shuffle:
        scans = sorted(scans)
        np.random.seed(12345)
    np.random.shuffle(scans)

    # now we put the images in tuples of matching patient ids
    scans_used = []
    scan_tuples = []
    patient_id_list = [patient_ids[scan] for scan in scans]
    for scan in scans:
        if scan not in scans_used:
            pat_id = patient_ids[scan]
            # find all scans with same patient id
            scan_tuple = [s for pid, s in zip(patient_id_list, scans) if
                          pid == pat_id]

            # save this tuple
            scan_tuples.append(scan_tuple)

            # and mark all the scans as used
            for s in scan_tuple:
                scans_used.append(s)

    # now we sort the tuples again to have the largest ones at first
    # this way we're trying to make the folds equally large. If a very large
    # tuple would be the last one in scan_tuples this might lead to very
    # unequally sized folds
    scan_tuples_len = [len(tpl) for tpl in scan_tuples]
    scan_tuples = [tpl for n, tpl in sorted(zip(scan_tuples_len, scan_tuples))]

    # now we unpack the guys and put them in folds
    folds = [[] for _ in range(n_folds)]
    for tpl in scan_tuples:
        # find out the fold with the currently lowest amount of scans
        ind = np.argmin([len(fold) for fold in folds])
        folds[ind].extend(tpl)

    # now we turn the folds into splits and return
    print('Smart splitting successfull length of folds:')
    print([len(fold) for fold in folds])
    print()

    return folds_to_splits(folds)
