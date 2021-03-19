from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.data.Dataset import Dataset
import os

model = SegmentationModel(val_fold=0,
                          data_name='OV04',
                          model_name='segmentation_on_Siemens_recons')

keys = ['image', 'label']
preprocessed_path = os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed',
                                 'OV04', 'pod_default')
model_names = ['recon_LPD_full_HU', 'recon_fbp_convs_full_HU', 'reconstruction_network_fbp_convs']

for model_name in model_names:
    scans = [scan for scan in os.listdir(os.path.join(preprocessed_path, model_name))
             if int(scan[5:8]) > 275]
    ds = Dataset(scans, preprocessed_path, keys, [model_name, 'labels'])
    model.eval_ds(ds, True, 'validation_'+model_name, False, False)
