from ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
import numpy as np
import torch
from ovseg.utils.torch_np_utils import maybe_add_channel_dim


class ClassSegmentationPreprocessing(SegmentationPreprocessing):

    def get_xb_from_data_tpl(self, data_tpl, get_only_im=False):
        
        # getting the image
        xb = data_tpl['image'].astype(float)

        # assuring the array is 4d
        xb = maybe_add_channel_dim(xb)

        if self.is_cascade():
            # the cascade is only implemented with binary predictions so far --> overwrite
            # this function for different predictions
            prev_preds = []
            for prev_stage, key in zip(self.prev_stages, self.keys_for_previous_stages):
                assert key in data_tpl, 'prediction '+key+' from previous stage missing'
                pred = data_tpl[key]
                if torch.is_tensor(pred):
                    pred = pred.cpu().numpy()
                # ensure the array is 4d
                pred = maybe_add_channel_dim(pred)
                
                if 'lb_classes' in prev_stage:
                    pred_new = np.zeros_like(pred)
                    for cl in prev_stage['lb_classes']:
                        pred_new[pred == cl] = 1
                    prev_preds.append(pred_new)
                else:
                    prev_preds.append((pred > 0).astype(float))
    
            bin_pred = np.max(np.stack(prev_preds), 0)

            xb = np.concatenate([xb, bin_pred])

        if 'label' in data_tpl and not get_only_im:     
            # get the label from the data_tpl and clean if applicable
            lb = self.maybe_clean_label_from_data_tpl(data_tpl)

            assert len(lb.shape) == 3, 'label must be 3d'
            lb = lb[np.newaxis].astype(float)
            xb = np.concatenate([xb, lb])
        
        # finally add batch axis
        xb = xb[np.newaxis]

        return xb
