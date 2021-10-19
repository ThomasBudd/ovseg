from ovseg.model.RegionexpertModel import RegionexpertModel
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from ovseg.model.RegionexpertEnsemble import RegionexpertEnsemble
from time import sleep

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
args = parser.parse_args()

w = 0.001

patch_size = [40, 320, 320]
use_prg_trn = True
out_shape = [[24, 192, 192],
             [28, 224, 224],
             [36, 288, 288],
             [40, 320, 320]]
larger_res_encoder = True


# %%
n_proc = 9
i = args.vf
all_lb_classes = []
for lb_classes in [[1], [9], [13, 15, 17], [2], [1, 2, 9]]:
    for _ in range(5):
        all_lb_classes.append(lb_classes)

all_vf = []
for _ in range(5):
    for vf in range(5):
        all_vf.append(vf)

vf_list = all_vf[i::n_proc]
lb_classes_list = all_lb_classes[i::n_proc]
print(vf_list)
print(lb_classes_list)
sleep(3)
# %%

for vf, lb_classes in zip(vf_list, lb_classes_list):

    model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                         z_to_xy_ratio=5.0/0.67,
                                                         use_prg_trn=use_prg_trn,
                                                         larger_res_encoder=larger_res_encoder,
                                                         n_fg_classes=1,
                                                         out_shape=out_shape)
    model_params['training']['loss_params'] = {'loss_names': ['cross_entropy',
                                                              'dice_loss']}
    model_params['data']['folders'] = ['images', 'labels', 'regions']
    model_params['data']['keys'] = ['image', 'label', 'region']
    model_params['training']['batches_have_masks'] = True
    model_params['postprocessing'] = {'mask_with_reg': True}
    model_params['data']['val_dl_params']['bias'] = 'cl_fg'
    model_params['data']['trn_dl_params']['bias'] = 'cl_fg'
    model_params['data']['val_dl_params']['n_fg_classes'] = len(lb_classes)
    model_params['data']['trn_dl_params']['n_fg_classes'] = len(lb_classes)
    
    model_name = 'U-Net5'
    p_name = 'reg_expert_'+'_'.join([str(c) for c in lb_classes])
    model = RegionexpertModel(val_fold=args.vf,
                              data_name='OV04',
                              preprocessed_name=p_name, 
                              model_name=model_name,
                              model_parameters=model_params)
    model.training.train()
    model.eval_validation_set()

lb_classes = lb_classes_list[args.vf]
p_name = 'reg_expert_'+'_'.join([str(c) for c in lb_classes])

ens = RegionexpertEnsemble(val_fold=list(range(5)),
                           data_name='OV04',
                           preprocessed_name=p_name, 
                           model_name=model_name)

while not ens.all_folds_complete():
    sleep(60)

ens.eval_raw_dataset('BARTS')