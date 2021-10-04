from ovseg.model.RegionfindingModel import RegionfindingModel
from ovseg.preprocessing.SLDSPreprocessing import SLDSPreprocessing
import numpy as np
from skimage.measure import label

class SLDSModel(RegionfindingModel):

    
    def _create_preprocessing_object(self):
        
        self.preprocessing = SLDSPreprocessing(**self.model_parameters['preprocessing'])

    def compute_error_metrics(self, data_tpl):
        if 'label' not in data_tpl:
            # in this case we're evaluating an unlabeled image so we can\'t compute any metrics
            return None
        pred = data_tpl[self.pred_key]
        # in case of raw data this only removes the lables that this model doesn't segment
        # seg = self.preprocessing.maybe_clean_label_from_data_tpl(data_tpl)
        # with the new update the prediction should be in classes as well instead of 
        # integer encoding as before. Let's hope that it works!
        seg = data_tpl['label']
        if len(seg.shape) == 4:
            seg = seg[0]

        seg = (seg > 0).astype(float)
        if self.preprocessing.is_preprocessed_data_tpl(data_tpl):
            region = data_tpl['region']
            if len(region.shape) == 4:
                region = region[0]
            lesions = region * seg
            small_lesions = np.zeros_like(seg)
            small_lesions[lesions > 1] = lesions[lesions > 1]
            large_lesions = (lesions == 1).astype(float)
        else:
            ccomps = label(seg)
            small_lesions = np.zeros_like(seg)
            large_lesions = np.zeros_like(seg)
            
            fac = np.prod(self.preprocessing.target_spacing)
            
            for c in range(1, ccomps.max() +1 ):
                comp = (ccomps == c)
                comp_vol = fac * np.sum(comp)
                
                reg_lb = self.preprocessing.determine_region_label(comp_vol)
                
                if reg_lb > 0:
                    small_lesions[comp] = reg_lb + 1
                else:
                    large_lesions[comp] = 1
                    
        # prec_0 is the amount of voxel that can be excluded with the method
        
        bin_seg = (seg > 0).astype(float)
        regions = (pred > 1).astype(float)
        bp_reg_seg = regions * bin_seg
        bp_final_seg = (pred == 1).astype(float) + bp_reg_seg
        
        bp_bin_dice = 200 * np.sum(bp_final_seg * bin_seg) / np.sum(bin_seg + bp_final_seg)
        
        results = {'bp_bin_dice': bp_bin_dice}
        
        large_pred = (pred == 1).astype(float)
        
        results['dice_large'] = 200 * np.sum(large_pred * large_lesions) / np.sum(large_lesions + large_pred)
        
        for c in range(2, self.preprocessing.n_fg_classes + 1):
            reg = (pred == c).astype(float)
            les = (small_lesions == c).astype(float)
            
            bp_les = reg * les
            results['bp_dice_{}'.format(c)] = 200 * np.sum(bp_les * les) / np.sum(bp_les + les)

        return results

    def _init_global_metrics(self):
        return

    def _update_global_metrics(self, data_tpl):

        return 