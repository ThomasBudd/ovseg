from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse
from time import sleep

parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
args = parser.parse_args()

p_name = 'lymph_nodes'
patch_sizes = [[40, 320, 320], [32, 256, 256]]
model_names = ['U-Net5', 'U-Net5_sp']
use_prg_trn = True
out_shapes = [[[24, 192, 192],
               [28, 224, 224],
               [36, 288, 288],
               [40, 320, 320]],
              [[20, 160, 160],
               [24, 192, 192],
               [28, 224, 224],
               [32, 256, 256]]]
larger_res_encoder = True
lb_classes = [13,14,15,17]

for patch_size, out_shape, model_name in zip(patch_sizes, out_shapes, model_names):

    model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                         z_to_xy_ratio=5.0/0.67,
                                                         use_prg_trn=use_prg_trn,
                                                         larger_res_encoder=larger_res_encoder,
                                                         n_fg_classes=len(lb_classes),
                                                         out_shape=out_shape)
    model_params['training']['loss_params'] = {'loss_names': ['cross_entropy',
                                                              'dice_loss']}
    
    model = SegmentationModel(val_fold=args.vf,
                              data_name='OV04',
                              preprocessed_name=p_name, 
                              model_name=model_name,
                              model_parameters=model_params)
    model.training.train()
    model.eval_raw_data_npz('BARTS')


    ens = SegmentationEnsemble(val_fold=list(range(5)),
                               data_name='OV04',
                               preprocessed_name=p_name, 
                               model_name=model_name)
    
    ens.wait_until_all_folds_complete()
    ens.eval_raw_dataset('BARTS')
