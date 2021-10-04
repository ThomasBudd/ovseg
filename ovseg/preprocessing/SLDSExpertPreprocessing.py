from ovseg.preprocessing.RegionexpertPreprocessing import RegionexpertPreprocessing
from ovseg.utils.seg_fg_dial import seg_fg_dial
import numpy as np
from ovseg.utils.label_utils import remove_small_connected_components_from_batch, reduce_classes, \
    remove_small_connected_components
from skimage.measure import label


class SLDSExpertPreprocessing(RegionexpertPreprocessing):

    def __init__(self, vol_tr=5000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if np.isscalar(vol_tr):
            self.vol_tr = [vol_tr]
        elif isinstance(vol_tr, np.ndarray):
            self.vol_tr = vol_tr.tolist()
        elif isinstance(vol_tr, tuple):
            self.vol_tr = list(vol_tr)
        elif isinstance(vol_tr, list):
            self.vol_tr = vol_tr
        else:
            raise TypeError('Got unrecognised input vol_tr of type {}, expceted int, float, '
                            'np.ndarray, list or tuple'.format(type(vol_tr)))

        self.vol_tr.append(0)
            
        self.preprocessing_parameters.append('vol_tr')
        
        self.n_fg_classes = len(self.vol_tr)

    def maybe_clean_label_from_data_tpl(self, data_tpl):

        if 'label' not in data_tpl:
            raise ValueError('Can\'t clean label from data tpl, none was found!')

        lb = data_tpl['label']

        if self.is_preprocessed_data_tpl(data_tpl):
            return lb

        lb = (lb > 0).astype(float)
        ccomps = label(lb)
        small_lesions = np.zeros_like(lb)
        
        fac = np.prod(self.target_spacing)
        
        for c in range(1, ccomps.max() +1 ):
            comp = (ccomps == c)
            comp_vol = fac * np.sum(comp)
            
            reg_lb = self.determine_region_label(comp_vol)
            
            if (reg_lb + 1) in self.lb_classes:
                small_lesions[comp] = 1


        return small_lesions


    def determine_region_label(self, vol):
        # computes which label the region will have:
        #   0: large lesions (no dialation)
        #   1: largest small lesions
        #   2: second largest small lesions... etc.
        
        for i in range(len(self.vol_tr) - 1):
            if self.vol_tr[i] > vol and vol > self.vol_tr[i+1]:
                return i+1

        return 0