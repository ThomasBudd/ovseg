from ovseg.data.SegmentationData import SegmentationData
from tqdm import tqdm
import os

ps = [32, 128, 128]
bs = 2
val_fold = 0
preprocessed_path = os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed', 'OV04', 'pod_half')
keys = ['image', 'label']
folders = ['images', 'labels']
trn_dl_params = {'patch_size': ps, 'batch_size': bs, 'store_coords_in_ram': True}

data = SegmentationData(val_fold, preprocessed_path, keys, folders, trn_dl_params=trn_dl_params)

n_fg = 0

for batch in tqdm(data.trn_dl):
    lb = batch[:, 1].cpu().numpy().astype(int)
    n_fg += lb.sum()

print('{:.2e}'.format(n_fg))
