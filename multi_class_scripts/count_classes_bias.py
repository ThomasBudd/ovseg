import numpy as np
from ovseg.data.Dataset import Dataset
from ovseg.data.SegmentationDataloader import SegmentationDataloader
from os import listdir, environ
from os.path import join

preprocessed_path = join(environ['OV_PREPROCESSED'], 'OV04', 'multiclass_1_2_9_13_15_17')
scans = listdir(join(preprocessed_path, 'images'))
keys = ['image', 'label']
folders = ['images', 'labels']

vol_ds = Dataset(scans, preprocessed_path, keys, folders)

patch_size = [40, 320, 320]
batch_size = 1


dl1 = SegmentationDataloader(vol_ds, patch_size, batch_size)
dl2 = SegmentationDataloader(vol_ds, patch_size, batch_size, bias='cl_fg', n_fg_classes=6)


def count_classes(dl, reps=4):
    f = np.zeros(6)
    for _ in range(reps):
        for batch in dl:
            batch = batch[:, -1].numpy()
            
            for b in range(batch_size):
                classes = np.array([c for c in np.unique(batch[b]) if c >0]) - 1
                f[classes.astype(int)] += 1
    return f

f1 = count_classes(dl1) / 10

f2 = count_classes(dl2) / 10

print('bias sampling: ' + ' '.join(['{:.1f}'.format(f) for f in f1]))
print('class bias sampling: ' + ' '.join(['{:.1f}'.format(f) for f in f2]))