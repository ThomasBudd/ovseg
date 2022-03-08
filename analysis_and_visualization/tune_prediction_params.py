from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.data.Dataset import raw_Dataset
from os.path import join
import os

model_name = 'gv_aug_5fCV_1_small_0'
p_name='pod_half'
ens = SegmentationEnsemble(val_fold=list(range(5)),
                            data_name='OV04',
                            preprocessed_name=p_name,
                            model_name=model_name)

ds = raw_Dataset(join(os.environ['OV_DATA_BASE'], 'raw_data', 'BARTS'))

# %% first let's do the constant patch weighting
for model in ens.models:
    model.model_parameters['prediction']['patch_weight_type'] = 'constant'
    model.initialise_prediction()
ens.eval_ds(ds, ds_name='BARTS_constant', save_preds=False)

# %% now gaussian
for sigma in [1/8, 1/4, 1/2]:
    for model in ens.models:
        model.model_parameters['prediction']['patch_weight_type'] = 'gaussian'
        model.model_parameters['prediction']['sigma_gaussian_weight'] = sigma
        model.initialise_prediction()
    ens.eval_ds(ds, ds_name='BARTS_gaussian_{:.3f}'.format(sigma), save_preds=False)

# %% now linear
for lin_min in [0.1, 0.5]:
    for model in ens.models:
        model.model_parameters['prediction']['patch_weight_type'] = 'linear'
        model.model_parameters['prediction']['linear_min'] = lin_min
        model.initialise_prediction()
    ens.eval_ds(ds, ds_name='BARTS_linear_{:.1f}'.format(lin_min), save_preds=False)
