import numpy as np


def eval_prediction_segmentation(seg, pred):

    n_classes = int(max(seg.max(), pred.max()))

    results = {}
    for c in range(1, n_classes+1):
        results = {}
        seg_c = (seg == c).astype(float)
        pred_c = (pred == c).astype(float)

        has_fg = seg_c.max() > 0
        fg_pred = pred_c.max() > 0

        results.update({'has_fg_%d' % c: seg_c.max() > 0,
                        'fg_pred_%d' % c: fg_pred})
        tp = np.sum(seg_c * pred_c)
        seg_c_vol = np.sum(seg_c)
        pred_c_vol = np.sum(pred_c)
        if has_fg and fg_pred:
            dice = 200 * tp / (seg_c_vol + pred_c_vol)
        else:
            dice = 100
        results.update({'dice_%d' % c: dice})

        if has_fg:
            sens = 100 * tp / seg_c_vol
        else:
            sens = np.nan

        if fg_pred:
            prec = 100 * tp / pred_c_vol
        else:
            prec = np.nan

        results.update({'sens_%d' % c: sens, 'prec_%d' % c: prec})
    return results
