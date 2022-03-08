from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.RegionexpertModel import RegionexpertModel
import numpy as np
import torch
from ovseg.utils.torch_np_utils import maybe_add_channel_dim
from os import environ
from os.path import join, exists

class RegionexpertEnsemble(SegmentationEnsemble):
    
    def create_model(self, fold):
        model = RegionexpertModel(val_fold=fold,
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
        scan = data_tpl['scan']
        if not self.preprocessing.is_preprocessed_data_tpl(data_tpl):
            # the image already contains the binary prediction as additional channel
            vol = self.preprocessing(data_tpl, preprocess_only_im=True)
            im, reg = vol[:-1], vol[-1:]
        else:
            # the data_tpl is already preprocessed, let's just get the arrays
            im = data_tpl['image']
            im = maybe_add_channel_dim(im)
            reg = data_tpl[self.prev_stages_keys[0]]
            reg = maybe_add_channel_dim(reg)

        # now the importat part: the actual enembling of sliding window evaluations
        preds = []
        # also the path where we will look for already executed npz prediction
        pred_npz_path = join(environ['OV_DATA_BASE'], 'npz_predictions', self.data_name,
                             self.preprocessed_name, self.model_name)
        with torch.no_grad():
            for model in self.models:
                # try find the npz file if there was already a prediction.
                path_to_npz = join(pred_npz_path, model.val_fold_str, scan+'.npz')
                path_to_npy = join(pred_npz_path, model.val_fold_str, scan+'.npy')
                if exists(path_to_npy):
                    pred = torch.from_numpy(np.load(path_to_npy)).to(self.dev)
                elif exists(path_to_npz):
                    pred = torch.from_numpy(np.load(path_to_npz)['arr_0']).to(self.dev)
                else:
                    pred = model.prediction(im, reg[0]>0)
                preds.append(pred.cpu().numpy())
            ens_pred = np.stack(preds).mean(0)
            data_tpl[self.pred_key] = ens_pred

        # inside the postprocessing the result will be attached to the data_tpl
        self.postprocessing.postprocess_data_tpl(data_tpl, self.pred_key, reg)

        torch.cuda.empty_cache()
        return data_tpl[self.pred_key]