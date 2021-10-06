import numpy as np
try:
    from skimage.measure import label
except ImportError:
    print('Caught Import Error while importing some function from scipy or skimage. '
          'Please use a newer version of gcc.')


def remove_small_connected_components(lb, min_vol, spacing=None):
    '''

    Parameters
    ----------
    lb : np.ndarray
        integer valued label array
    min_vol : scalar or list
        smallest volume
    spacing : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    lb : TYPE
        DESCRIPTION.

    '''

    n_classes = int(lb.max())
    if np.isscalar(min_vol):
        min_vol = [min_vol for _ in range(n_classes)]
    else:
        if len(min_vol) < n_classes:
            raise ValueError('min_vol was given as a list, but with less volumes then classes. '
                             'Choose either one univsersal number for all classes or give '
                             'at least as many volumes as classes.')

    if spacing is None:
        spacing = [1, 1, 1]

    fac = np.prod(spacing)

    for i, mvol in enumerate(min_vol):
        bin_label = (lb == i + 1) > 0

        conn_comps = label(bin_label)
        n_comps = conn_comps.max()

        for c in range(1, n_comps+1):
            conn_comp = (conn_comps == c)

            if np.sum(conn_comp.astype(float)) * fac < mvol:
                lb[conn_comp] = 0

    return lb

def remove_connected_components_by_volume(lb, min_vol=None, max_vol=None, spacing=None):
    '''

    Parameters
    ----------
    lb : np.ndarray
        integer valued label array
    min_vol : scalar or list
        smallest volume
    max_vol : scalar or list
        largest volume
    spacing : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    lb : TYPE
        DESCRIPTION.

    '''

    n_classes = int(lb.max())
    
    if min_vol is None:
        min_vol = 0
    if max_vol is None:
        max_vol = np.inf
    
    if np.isscalar(min_vol):
        min_vol = [min_vol for _ in range(n_classes)]
    else:
        if len(min_vol) < n_classes:
            raise ValueError('min_vol was given as a list, but with less volumes then classes. '
                             'Choose either one univsersal number for all classes or give '
                             'at least as many volumes as classes.')
    if np.isscalar(max_vol):
        max_vol = [max_vol for _ in range(n_classes)]
    else:
        if len(max_vol) < n_classes:
            raise ValueError('min_vol was given as a list, but with less volumes then classes. '
                             'Choose either one univsersal number for all classes or give '
                             'at least as many volumes as classes.')

    if spacing is None:
        spacing = [1, 1, 1]

    fac = np.prod(spacing)

    for i, (mnvol, mxvol) in enumerate(zip(min_vol, max_vol)):
        bin_label = (lb == i + 1) > 0

        conn_comps = label(bin_label)
        n_comps = conn_comps.max()

        for c in range(1, n_comps+1):
            conn_comp = (conn_comps == c)

            conn_comp_vol = np.sum(conn_comp.astype(float)) * fac
            if conn_comp_vol < mnvol:
                lb[conn_comp] = 0
            elif conn_comp_vol > mxvol:
                lb[conn_comp] = 0

    return lb


def remove_small_connected_components_from_batch(lbb, min_vol, spacing=None):

    batch_list = []
    for b in range(lbb.shape[0]):
        channel_list = []
        for c in range(lbb.shape[1]):
            lb = remove_small_connected_components(lbb[b, c], min_vol, spacing)
            channel_list.append(lb)
        batch_list.append(np.stack(channel_list))
    return np.stack(batch_list)

def remove_connected_components_by_volume_from_batch(lbb, min_vol=None, max_vol=None, spacing=None):

    batch_list = []
    for b in range(lbb.shape[0]):
        channel_list = []
        for c in range(lbb.shape[1]):
            lb = remove_connected_components_by_volume(lbb[b, c], min_vol, max_vol, spacing)
            channel_list.append(lb)
        batch_list.append(np.stack(channel_list))
    return np.stack(batch_list)


def reduce_classes(lb, classes, to_single_class=False):

    lb_new = np.zeros_like(lb)

    for i, c in enumerate(classes):
        lb_new[lb == c] = i + 1

    if to_single_class:
        lb_new = (lb_new > 0).astype(lb.dtype)

    return lb_new
