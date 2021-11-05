from ovseg.data.ClassSegmentationDataloader import ClassSegmentationBatchDataset
import numpy as np
import torch
import os

class ClassCascadeBatchDataset(ClassSegmentationBatchDataset):

    def __getitem__(self, index):
        
        volume = super().__getitem__(index)
        
        # prediction from previous stage and binarise it
        bin_pred = (volume[-2:-1] > 0).astype(volume.dtype)
        mask = 1 - bin_pred
        return np.concatenate([volume[:-2], mask, volume[-1:]])


def ClassCascadeDataloader(vol_ds, patch_size, batch_size, num_workers=None,
                           pin_memory=True, epoch_len=250, *args, **kwargs):
    
    kwargs['batches_have_masks'] = False
    
    dataset = ClassCascadeBatchDataset(vol_ds=vol_ds,
                                       patch_size=patch_size,
                                       batch_size=batch_size,
                                       epoch_len=epoch_len,
                                       *args, **kwargs)
    if num_workers is None:
        num_workers = 0 if os.name == 'nt' else 8
    worker_init_fn = lambda _: np.random.seed()
    sampler = torch.utils.data.SequentialSampler(range(batch_size * epoch_len))
    return torch.utils.data.DataLoader(dataset,
                                       sampler=sampler,
                                       batch_size=batch_size,
                                       pin_memory=pin_memory,
                                       num_workers=num_workers,
                                       worker_init_fn=worker_init_fn)
