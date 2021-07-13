from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.data.RegionfindingData import RegionfindingData
import numpy as np

class RegionfindingModel(SegmentationModel):

    def initialise_data(self):
        # the data object holds the preprocessed data (training and validation)
        # for each it has both a dataset returning the data tuples and the dataloaders
        # returning the batches
        if 'data' not in self.model_parameters:
            raise AttributeError('model_parameters must have key '
                                 '\'data\'. These must contain the '
                                 'dict of training paramters.')

        # Let's get the parameters and add the cpu augmentation
        params = self.model_parameters['data'].copy()

        # if we don't want to store our data in ram...
        if self.dont_store_data_in_ram:
            for key in ['trn_dl_params', 'val_dl_params']:
                params[key]['store_data_in_ram'] = False
                params[key]['store_coords_in_ram'] = False
        self.data = RegionfindingData(val_fold=self.val_fold,
                                      preprocessed_path=self.preprocessed_path,
                                      augmentation= self.augmentation.np_augmentation,
                                      **params)
        print('Data initialised')


    def compute_error_metrics(self, data_tpl):
        if 'label' not in data_tpl:
            # in this case we're evaluating an unlabeled image so we can\'t compute any metrics
            return None
        pred = data_tpl[self.pred_key]
        # in case of raw data this only removes the lables that this model doesn't segment
        seg = self.preprocessing.maybe_clean_label_from_data_tpl(data_tpl)
        bin_seg = (seg > 0).astype(float)
        results = {'prec_0': np.mean(pred == 0)}
        for c in range(1, self.n_fg_classes+1):
            seg_c = (seg == c).astype(float)
            pred_c = (pred == c).astype(float)
            pred_c_mask = pred_c * bin_seg

            tp = np.sum(seg_c * pred_c_mask)
            seg_c_vol = np.sum(seg_c)
            pred_c_vol = np.sum(pred_c_mask)
            if seg_c_vol > 0:
                dice = 200 * tp / (seg_c_vol + pred_c_vol)
            else:
                dice = np.nan
            results.update({'bp_dice_%d' % c: dice, 'prec_%d' % c: np.mean(pred_c)})

        return results

    def _init_global_metrics(self):
        self.global_metrics_helper = {'total_vol': 0}
        self.global_metrics = {}
        for c in range(1, self.n_fg_classes + 1):
            self.global_metrics_helper.update({s+str(c): 0 for s in ['vol_']})
            self.global_metrics.update({'prec_'+str(c): 0})

    def _update_global_metrics(self, data_tpl):

        pred = data_tpl[self.pred_key]

        # volume of one voxel
        fac = np.prod(data_tpl['spacing']) if 'spacing' in data_tpl else 1
        self.global_metrics_helper['total_vol'] += np.prod(pred.shape) * fac
        tv = self.global_metrics_helper['total_vol']
        for c in range(1, self.n_fg_classes + 1):
            pred_c = (pred == c).astype(float)
            self.global_metrics_helper['vol_'+str(c)] += np.sum(pred_c) * fac
            # now update global metrics helper
            self.global_metrics['prec_'+str(c)] = self.global_metrics_helper['vol_'+str(c)] / tv
