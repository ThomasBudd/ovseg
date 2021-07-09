from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("fold", type=int)
parser.add_argument("exp", type=int)
args = parser.parse_args()


p_names = [['om_mask', 'lesions_upper_mask'], ['lesions_center_mask', 'lesions_lymphnodes_mask'],
          ['pod_mask']][args.exp]

if args.exp < 2:
    use_prg_trn = False
    model_name = 'res_encoder'
    patch_size = [32, 216, 216]
    out_shape = None
    larger_res_encoder = False
    z_to_xy_ratio = 6.25
    fold_list = [args.fold]
elif args.exp == 2:
    fold_list = list(range(5))
    use_prg_trn = True
    model_name = 'larger_res_encoder'

    patch_size = [32, 256, 256]
    out_shape = [[20, 160, 160],
                 [24, 192, 192],
                 [28, 224, 224],
                 [32, 256, 256]]
    larger_res_encoder = True
    z_to_xy_ratio = 5.0/0.67
        

model_params = get_model_params_3d_res_encoder_U_Net(patch_size, z_to_xy_ratio, use_prg_trn,
                                                     out_shape=out_shape,
                                                     larger_res_encoder=larger_res_encoder)

# now make sure that the masks are used!
model_params['data']['folders'].append('masks')
model_params['data']['keys'].append('mask')

model_params['data']['trn_dl_params']['mask_key'] = 'mask'
model_params['data']['val_dl_params']['mask_key'] = 'mask'
model_params['training']['batches_have_masks'] = True
model_parsms['training']['prg_trn_resize_on_the_fly'] = True

for p_name, fold in zip(p_names, fold_list):
    model = SegmentationModel(val_fold=fold,
                              data_name='OV04',
                              preprocessed_name=p_name,
                              model_name=model_name,
                              model_parameters=model_params)
    model.training.train()
    model.eval_validation_set()
    model.eval_raw_data_npz('BARTS')
    model.clean()
