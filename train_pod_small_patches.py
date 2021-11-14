from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
import argparse
from time import sleep

parser = argparse.ArgumentParser()
parser.add_argument("vf", type=int)
args = parser.parse_args()

p_name='pod_067'

patch_size = [32, 256, 256]
use_prg_trn = True
out_shape = [[20, 160, 160], [24, 192, 192], [28, 224, 224], [32, 256, 256]]
larger_res_encoder = True

architectures = ['unetresencoder', 'unetresstemencoder']

for architecture in architectures:

    model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                          z_to_xy_ratio=5.0/0.67,
                                                          use_prg_trn=use_prg_trn,
                                                          larger_res_encoder=larger_res_encoder,
                                                          n_fg_classes=1,
                                                          out_shape=out_shape)
    model_params['training']['loss_params'] = {'loss_names': ['cross_entropy',
                                                               'dice_loss']}
    
    model_params['architecture'] = architecture
    model = SegmentationModel(val_fold=args.vf,
                              data_name='OV04',
                              preprocessed_name=p_name,
                              model_name=architecture + '_sp',
                              model_parameters=model_params)
    
    model.training.train()
    model.eval_raw_data_npz('BARTS')
    
    ens = SegmentationEnsemble(val_fold=list(range(5,8)),
                                 data_name='OV04',
                                 preprocessed_name=p_name,
                                 model_name=architecture + '_sp')
    
    while not ens.all_folds_complete():
        sleep(20)
    
    ens.eval_raw_dataset('BARTS')