from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.model_parameters_segmentation import get_model_params_2d_segmentation
import torch
from ovseg.training.SegmentationTraining import SegmentationTraining

model_parameters = get_model_params_2d_segmentation(batch_size=3)
model_parameters['training']['num_epochs'] = 5
model_parameters['data']['trn_dl_params']['epoch_len'] = 25
model_parameters['data']['val_dl_params']['epoch_len'] = 2

model = SegmentationModel(val_fold=0, data_name='OV04', model_parameters=model_parameters,
                          model_name='test_val_time_training')

# first do the usual training with default options
model.training.train()

# %%
dl = model.data.val_dl

val_dl_new = torch.utils.data.DataLoader(dl.dataset,
                                         sampler=dl.sampler,
                                         batch_size=dl.batch_size,
                                         pin_memory=dl.pin_memory,
                                         num_workers=dl.num_workers)
params = model.model_parameters['training'].copy()
params['num_epochs'] = 10
training_new = SegmentationTraining(network=model.network,
                                    trn_dl=model.data.trn_dl,
                                    val_dl=val_dl_new,
                                    model_path=model.model_path,
                                    network_name=model.network_name,
                                    augmentation=model.augmentation.GPU_augmentation,
                                    **params)
training_new.train()
