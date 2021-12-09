from ovseg.prerpocessing.SegmentationPreprocessing import SegmentationPreprocessing
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.SegmentationModel import SegmentationModel
import os
from ovseg.model.model_parameters_segmentation import get_model_params_3d_res_encoder_U_Net
from ovseg import OV_PREPROCESSED

data_name = 'OV04'
p_name = 'abd_les_bin'
model_name = 'U-Net4'

ppath = os.path.join(OV_PREPROCESSED, data_name, p_name)

if not os.path.exists(ppath):
    
    prep = SegmentationPreprocessing(apply_resizing=True,
                                     apply_pooling=False,
                                     apply_windowing=True,
                                     lb_classes=[1,2,3,4,5,6,7],
                                     save_only_fg_scans=True,
                                     reduce_lb_to_single_class=True)
    
    prep.plan_preprocessing_raw_data(data_name)
    
    prep.preprocess_raw_data(data_name, p_name)

patch_size = [32, 216, 216]
use_prg_trn = True
out_shape = [[20, 128, 128],
             [22, 152, 152],
             [30, 192, 192],
             [32, 216, 216]]
larger_res_encoder = False

model_params = get_model_params_3d_res_encoder_U_Net(patch_size=patch_size,
                                                     z_to_xy_ratio=5.0/0.8,
                                                     use_prg_trn=use_prg_trn,
                                                     larger_res_encoder=larger_res_encoder,
                                                     n_fg_classes=1,
                                                     out_shape=out_shape)
for vf in [5,6,7]:
    model = SegmentationModel(val_fold=vf,
                              data_name='OV04',
                              preprocessed_name=p_name, 
                              model_name=model_name,
                              model_parameters=model_params)
    model.training.train()
    model.clean()
    
ens = SegmentationEnsemble(val_fold=list(range(5,8)),
                           data_name='OV04',
                           preprocessed_name=p_name, 
                           model_name=model_name)


ens.eval_raw_dataset('BARTS')
