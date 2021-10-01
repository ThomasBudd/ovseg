from ovseg.preprocessing.RegionfindingPreprocessing import RegionfindingPreprocessing
from ovseg.utils.seg_fg_dial import seg_fg_dial
from skimage.measure import label
import numpy as np


class SLDSPreprocessing(RegionfindingPreprocessing):

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
        
        self.n_fg_classes = len(self.vol_tr) + 1

    def seg_to_region(self, seg):
        
        seg = (seg > 0).astype(float)
        ccomps = label(seg)
        small_lesions = np.zeros_like(seg)
        large_lesions = np.zeros_like(seg)
        
        fac = np.prod(self.target_spacing)
        
        for c in range(1, ccomps.max() +1 ):
            comp = (ccomps == c)
            comp_vol = fac * np.sum(comp)
            
            reg_lb = self.determine_region_label(comp_vol)
            
            if reg_lb > 0:
                small_lesions[comp] = reg_lb
            else:
                large_lesions[comp] = 1

        # we encode the small lesions as 0 background, 1, 2,... etc. 
        regions = seg_fg_dial(small_lesions, r=self.r, z_to_xy_ratio=self.z_to_xy_ratio)
        
        # change that to small lesions being 2, 3, ... and 1 being large lesions
        regions[regions>0] += 1
        regions[large_lesions > 0] = 1

        return regions

    def determine_region_label(self, vol):
        # computes which label the region will have:
        #   0: large lesions (no dialation)
        #   1: largest small lesions
        #   2: second largest small lesions... etc.
        
        for i in range(len(self.vol_tr) - 1):
            if self.vol_tr[i] > vol and vol > self.vol_tr[i+1]:
                return i+1

        return 0