from ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
from ovseg.utils.torch_np_utils import maybe_add_channel_dim
import torch
import numpy as np

class ClassCascadePreprocessing(SegmentationPreprocessing):
    
    def __init__(self, *args, prev_pred_classes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_pred_classes = prev_pred_classes
        self.preprocessing_parameters.append('prev_pred_classes')
    
    def is_cascade(self):
        assert len(self.prev_stages) == 1, 'in a cascade we need exactly one previous model'
        return True        
    
    def get_xb_from_data_tpl(self, data_tpl, get_only_im=False):
        
        # getting the image
        xb = data_tpl['image'].astype(float)

        # assuring the array is 4d
        xb = maybe_add_channel_dim(xb)

        if self.is_cascade():
            key = self.keys_for_previous_stages[0]
            assert key in data_tpl, 'prediction '+key+' from previous stage missing'
            pred = data_tpl[key]
            if torch.is_tensor(pred):
                pred = pred.cpu().numpy()
            # ensure the array is 4d
            pred = maybe_add_channel_dim(pred)
            
            if self.prev_pred_classes is not None:
                pp_classes = [c for c in np.unique(pred) if c > 0]
                ignor_this_classes = [c for c in pp_classes if c not in self.prev_pred_classes]
                for c in ignor_this_classes:
                    pred[pred == c] = 0

            xb = np.concatenate([xb, pred])

        if 'label' in data_tpl and not get_only_im:     
            # get the label from the data_tpl and clean if applicable
            lb = self.maybe_clean_label_from_data_tpl(data_tpl)

            assert len(lb.shape) == 3, 'label must be 3d'
            lb = lb[np.newaxis].astype(float)
            xb = np.concatenate([xb, lb])
        
        # finally add batch axis
        xb = xb[np.newaxis]

        return xb