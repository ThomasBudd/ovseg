from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.SegmentationModelV2 import SegmentationModelV2
from ovseg.data.Dataset import raw_Dataset
from os.path import join, exists
from os import environ
import torch
import numpy as np


class SegmentationEnsembleV2(SegmentationEnsemble):
    
    def create_model(self, fold):
        model = SegmentationModelV2(val_fold=fold,
                                    data_name=self.data_name,
                                    model_name=self.model_name,
                                    model_parameters=self.model_parameters,
                                    preprocessed_name=self.preprocessed_name,
                                    network_name=self.network_name,
                                    is_inference_only=True,
                                    fmt_write=self.fmt_write,
                                    model_parameters_name=self.model_parameters_name
                                    )
        return model
       
    def __call__(self, data_tpl):
        if not self.all_folds_complete():
            print('WARNING: Ensemble is used without all training folds being completed!!')
        
        if not self.models_initialised:
            print('Models were not initialised. Trying to do it now...')
            self.wait_until_all_folds_complete()
        
        scan = data_tpl['scan']

        # also the path where we will look for already executed npz prediction
        pred_npz_path = join(environ['OV_DATA_BASE'], 'npz_predictions', self.data_name,
                             self.preprocessed_name, self.model_name)
        
        # the preprocessing will only do something if the image is not preprocessed yet
        if not self.preprocessing.is_preprocessed_data_tpl(data_tpl):
            for model in self.models:
                # try find the npz file if there was already a prediction.
                path_to_npz = join(pred_npz_path, model.val_fold_str, scan+'.npz')
                path_to_npy = join(pred_npz_path, model.val_fold_str, scan+'.npy')
                
                if exists(path_to_npy) or exists(path_to_npz):
                    im, mask = None, None
                    continue
                else:
                    im = self.preprocessing(data_tpl, preprocess_only_im=True)
                    if self.preprocessing.has_ps_mask:
                        im, mask = im[:-1], im[-1:]
                    else:
                        mask = None
                    break

        # now the importat part: the actual enembling of sliding window evaluations
        preds = []
        with torch.no_grad():
            for model in self.models:
                # try find the npz file if there was already a prediction.
                path_to_npz = join(pred_npz_path, model.val_fold_str, scan+'.npz')
                path_to_npy = join(pred_npz_path, model.val_fold_str, scan+'.npy')
                if exists(path_to_npy):
                    try:
                        pred = np.load(path_to_npy)
                    except ValueError:
                        
                        if im is None:
                            im = self.preprocessing(data_tpl, preprocess_only_im=True)
                            if self.preprocessing.has_ps_mask:
                                im, mask = im[:-1], im[-1:]
                            else:
                                mask = None
                        pred = model.prediction(im).cpu().numpy()
                elif exists(path_to_npz):
                    try:
                        pred = np.load(path_to_npz)['arr_0']
                    except ValueError:
                        if im is None:
                            im = self.preprocessing(data_tpl, preprocess_only_im=True)
                            if self.preprocessing.has_ps_mask:
                                im, mask = im[:-1], im[-1:]
                            else:
                                mask = None
                        pred = model.prediction(im).cpu().numpy()
                        
                else:
                    pred = model.prediction(im).cpu().numpy()
                preds.append(pred)
            
            ens_pred = np.stack(preds).mean(0)
                
            data_tpl[self.pred_key] = ens_pred

        # inside the postprocessing the result will be attached to the data_tpl
        self.postprocessing.postprocess_data_tpl(data_tpl, self.pred_key, mask)

        torch.cuda.empty_cache()
        return data_tpl[self.pred_key]

    def eval_raw_dataset(self, data_name, save_preds=True, save_plots=False,
                         force_evaluation=False, scans=None, image_folder=None, dcm_revers=True,
                         dcm_names_dict=None):
        
        prev_stages = {**self.preprocessing.prev_stage_for_input,
                       **self.preprocessing.prev_stage_for_mask}
        if len(prev_stages) == 0:
            prev_stages = None
        
        ds = raw_Dataset(join(environ['OV_DATA_BASE'], 'raw_data', data_name),
                         scans=scans,
                         image_folder=image_folder,
                         dcm_revers=dcm_revers,
                         dcm_names_dict=dcm_names_dict,
                         prev_stages=prev_stages)
        self.eval_ds(ds, ds_name=data_name, save_preds=save_preds, save_plots=save_plots,
                     force_evaluation=force_evaluation)