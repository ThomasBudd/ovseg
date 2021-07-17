from ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
from ovseg.utils.label_utils import remove_small_connected_components_from_batch, reduce_classes, \
    remove_small_connected_components
import numpy as np
import torch

class RegionexpertPreprocessing(SegmentationPreprocessing):

    def __init__(self, *args, region_finding_model, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.region_finding_model = region_finding_model
        
        for key in ['data_name', 'preprocessed_name', 'model_name']:
            assert key in self.region_finding_model
        self.region_finding_key = '_'.join(['prediction',
                                            self.region_finding_model['data_name'],
                                            self.region_finding_model['preprocessed_name'],
                                            self.region_finding_model['model_name']])

        self.preprocessing_parameters = ['apply_resizing',
                                         'apply_pooling',
                                         'apply_windowing',
                                         'target_spacing',
                                         'pooling_stride',
                                         'window',
                                         'scaling',
                                         'lb_classes',
                                         'mask_calsses',
                                         'reduce_lb_to_single_class',
                                         'lb_min_vol',
                                         'n_im_channels',
                                         'do_nn_img_interp',
                                         'save_only_fg_scans',
                                         'prev_stages',
                                         'dataset_properties',
                                         'region_finding_model']

    
    def get_xb_from_data_tpl(self, data_tpl, get_only_im=False):
        
        xb = super().get_xb_from_data_tpl(data_tpl, get_only_im)
        if get_only_im:
            return xb

        # in contrast to the previous method we need to get the region as well and might merge it
        # with the other masks
        assert self.region_finding_key in data_tpl, 'Regions not found in data_tpl'
        region = data_tpl[self.region_finding_key]
        if self.lb_classes is None:
            mask = region > 0
        else:
            mask = np.zeros_like(region)
            for c in self.lb_classes:
                mask[region == c] = 1

        if self.use_masks():
            xb[-2:-1] = mask * xb[-2:-1]
        else:
            xb = np.concatenate([xb[:-1], mask, xb[-1:]])

        return xb

    def slice_xb_to_dict(self, xb):

        d = {}
        im = xb[:self.n_im_channels]
        d['image'] = im
        lb = xb[-1].astype(np.uint8)
        d['label'] = lb
        if self.is_cascade():
            bin_pred = xb[self.n_im_channels].astype(np.uint8)
            d['bin_pred'] = bin_pred
        mask = xb[-2].astype(np.uint8)
        d['mask'] = mask
        return d
        