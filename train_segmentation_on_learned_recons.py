from ovseg.model.model_parameters_segmentation import get_model_params_2d_segmentation
from ovseg.model.SegmentationModel import SegmentationModel
import torch

val_fold = 0
data_name = 'OV04'

# %% first segmentation on siemens reconstructions
model_params = get_model_params_2d_segmentation()

model = SegmentationModel(val_fold=val_fold, data_name=data_name, model_parameters=model_params,
                          model_name='segmentation_on_Siemens_recons')
model.training.train()
model.eval_validation_set()
model.eval_training_set()
torch.cuda.empty_cache()


# %%
model_name = 'recon_fbp_convs_high_0_long'
#['recon_LPD_full_HU', 'recon_fbp_convs_full_HU', 'reconstruction_network_fbp_convs']

for i in range(5):
    model_params = get_model_params_2d_segmentation()
    model_params['data']['folders'][0] = model_name

    model = SegmentationModel(val_fold=val_fold, data_name=data_name, model_parameters=model_params,
                              model_name='segmentation_on_'+model_name+'_'+str(i))
    model.training.train()
    model.eval_validation_set()
    model.eval_training_set()
    torch.cuda.empty_cache()
