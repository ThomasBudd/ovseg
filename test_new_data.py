from ovseg.data.SegmentationData import SegmentationData
from ovseg.model.model_parameters_segmentation import get_model_params_2d_segmentation
import os
import numpy as np
from tqdm import tqdm
model_params = get_model_params_2d_segmentation()
data_params = model_params['data']
data_params['trn_dl_params']['store_coords_in_ram'] = True
data_params['val_dl_params']['store_coords_in_ram'] = False
val_fold = 0
preprocessed_path = os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed',
                                 'OV04_BARTS_ApolloTCGA', 'default')
data = SegmentationData(val_fold=val_fold,
                        preprocessed_path=preprocessed_path,
                        **data_params)
# %%
means = []
stds = []
for batch in tqdm(data.trn_dl):
    im = batch.cpu().numpy()[0, :, 0]
    means.extend(np.mean(im, (1, 2)).tolist())
    stds.extend(np.std(im, (1, 2)).tolist())
