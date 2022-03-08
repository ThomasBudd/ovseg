from ovseg.model.ClassCascadeModel import ClassCascadeModel
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from os.path import join, exists
from os import environ
import torch
import numpy as np
from ovseg.utils.torch_np_utils import maybe_add_channel_dim


class ClassCascadeEnsemble(SegmentationEnsemble):
    
    def create_model(self, fold):
        model = ClassCascadeModel(val_fold=fold,
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
    
    def _get_im_prev_pred_from_data_tpl(self, data_tpl):
        if not self.preprocessing.is_preprocessed_data_tpl(data_tpl):
            # the image already contains the binary prediction as additional channel
            volume = self.preprocessing(data_tpl, preprocess_only_im=True, return_np=True)
            im, prev_pred = volume[:-1], volume[-1:]
        else:
            # the data_tpl is already preprocessed, let's just get the arrays
            im = maybe_add_channel_dim(data_tpl['image'])
            prev_pred = maybe_add_channel_dim(data_tpl['prev_pred'])

        # the input to the network (predicion object) must be with binarised
        # previous prediction
        # im = np.concatenate([im, (prev_pred > 0).astype(im.dtype)])
        
        return im, prev_pred 

    def __call__(self, data_tpl):
        if not self.all_folds_complete():
            print('WARNING: Ensemble is used without all training folds being completed!!')
        scan = data_tpl['scan']

        # also the path where we will look for already executed npz prediction
        pred_npz_path = join(environ['OV_DATA_BASE'], 'npz_predictions', self.data_name,
                             self.preprocessed_name, self.model_name)
        
        # the preprocessing will only do something if the image is not preprocessed yet
        im, prev_pred = self._get_im_prev_pred_from_data_tpl(data_tpl)

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
                        
                        pred = model.prediction(im).cpu().numpy()
                elif exists(path_to_npz):
                    try:
                        pred = np.load(path_to_npz)['arr_0']
                    except ValueError:
                        pred = model.prediction(im).cpu().numpy()
                else:
                    pred = model.prediction(im).cpu().numpy()
                preds.append(pred)
            ens_pred = np.stack(preds).mean(0)
            data_tpl[self.pred_key] = ens_pred

        # inside the postprocessing the result will be attached to the data_tpl
        self.postprocessing.postprocess_data_tpl(data_tpl, self.pred_key, prev_pred)

        torch.cuda.empty_cache()
        return data_tpl[self.pred_key]