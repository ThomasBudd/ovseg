from ovseg.data.Dataset import raw_Dataset
from ovseg.utils.label_utils import reduce_classes
import numpy as np
from tqdm import tqdm

lb_classes = [13,14,15,17]

prev_stages = [{'data_name': 'OV04',
                'preprocessed_name': 'lymph_nodes',
                'model_name': 'U-Net5_sp'}]

ds = raw_Dataset('BARTS', prev_stages=prev_stages)
key = ds.keys_for_previous_stages[0]
dscs = []
fps = []

for i in tqdm(range(len(ds))):
    data_tpl = ds[i]
    
    bin_lb = (reduce_classes(data_tpl['label'], lb_classes) > 0).astype(float)
    bin_pred = (data_tpl[key] > 0).astype(float)

    if bin_lb.max():
        dscs.append(200 * np.sum(bin_lb * bin_pred)/np.sum(bin_lb+bin_pred))
    else:
        fps.append(float(bin_pred.max() > 0))


print(dscs)
print(np.mean(dscs))
print(100*np.mean(fps))