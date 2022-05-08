from ovseg.utils.io import load_pkl
from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.ModelBase import ModelBase
from ovseg.data.Dataset import raw_Dataset
from os import environ, listdir
from os.path import join, isdir, exists
import torch
from ovseg.utils.torch_np_utils import check_type
import numpy as np
from tqdm import tqdm
from time import sleep


class SegmentationEnsemble(ModelBase):
    '''
    Ensembling Model that is used to add over softmax outputs before applying the argmax
    It is always called in inference mode!
    '''

    def __init__(self, data_name: str, model_name: str, preprocessed_name: str, val_fold=None,
                 network_name='network', fmt_write='{:.4f}',
                 model_parameters_name='model_parameters'):
        self.model_cv_path = join(environ['OV_DATA_BASE'],
                                  'trained_models',
                                  data_name,
                                  preprocessed_name,
                                  model_name)
        if val_fold is None:
            fold_folders = [f for f in listdir(self.model_cv_path)
                            if isdir(join(self.model_cv_path, f)) and f.startswith('fold')]
            val_fold = [int(f.split('_')[-1]) for f in fold_folders]
        super().__init__(val_fold=val_fold, data_name=data_name, model_name=model_name,
                         preprocessed_name=preprocessed_name,
                         network_name=network_name, is_inference_only=True,
                         fmt_write=fmt_write, model_parameters_name=model_parameters_name)

        # create all models
        self.models = []


        self.models_initialised = False
        if self.all_folds_complete():       
            self.initialise_models()

    def create_model(self, fold):
        model = SegmentationModel(val_fold=fold,
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

    def initialise_models(self):
        
        if self.models_initialised:
            print('Models were already initialised')
            return
        
        not_finished_folds = self._find_incomplete_folds()
        for fold in self.val_fold:
            if fold in not_finished_folds:
                print('Skipping fold {}. Training was not finished.'.format(fold))
                continue
            print('Creating model from fold: '+str(fold))
            model = self.create_model(fold)
            self.models.append(model)

        # change in evaluation mode
        for model in self.models:
            model.network.eval()
        
        self.models_initialised = True
        
        # now we do a hack by initialising the two objects like this...
        self.preprocessing = self.models[0].preprocessing
        self.postprocessing = self.models[0].postprocessing

        self.n_fg_classes = self.models[0].n_fg_classes
        if self.is_cascade():
            self.prev_stages = self.model_parameters['prev_stages'] 
            self.prev_stages_keys = []
            for prev_stage in self.prev_stages:
                key = '_'.join(['prediction',
                                prev_stage['data_name'],
                                prev_stage['preprocessed_name'],
                                prev_stage['model_name']])
                self.prev_stages_keys.append(key)

    def is_cascade(self):
        return 'prev_stages' in self.model_parameters

    def _find_incomplete_folds(self):
        num_epochs = self.model_parameters['training']['num_epochs']
        not_finished_folds = []
        for fold in self.val_fold:
            path_to_attr = join(self.model_cv_path,
                                'fold_'+str(fold),
                                'attribute_checkpoint.pkl')
            if not exists(path_to_attr):
                print('Trying to check if the training is done for all folds,'
                      ' but not checkpoint was found for fold '+str(fold)+'.')
                not_finished_folds.append(fold)
                continue

            attr = load_pkl(path_to_attr)

            if attr['epochs_done'] < attr['num_epochs']:
                not_finished_folds.append(fold)
        return not_finished_folds

    def all_folds_complete(self):
        not_finished_folds = self._find_incomplete_folds()
        if len(not_finished_folds) == 0:
            return True

        else:
            print("It seems like the folds " + str(not_finished_folds) +
                  " have not finished training.")
            return False
    
    def wait_until_all_folds_complete(self):
        
        waited = 0
        while not self.all_folds_complete():
            sleep(60)
            waited += 60
            
            if waited % 600 == 0:
                print('Waited {} seconds'.format(waited))
        
        self.initialise_models()

    def initialise_preprocessing(self):
        return

    def initialise_augmentation(self):
        return

    def initialise_network(self):
        return

    def initialise_postprocessing(self):
        return

    def initialise_data(self):
        return

    def initialise_training(self):
        return

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
                    im = None
                    continue
                else:
                    im = self.preprocessing(data_tpl, preprocess_only_im=True)
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
                        pred = model.prediction(im).cpu().numpy()
                elif exists(path_to_npz):
                    try:
                        pred = np.load(path_to_npz)['arr_0']
                    except ValueError:
                        if im is None:
                            im = self.preprocessing(data_tpl, preprocess_only_im=True)
                        pred = model.prediction(im).cpu().numpy()
                        
                else:
                    pred = model.prediction(im).cpu().numpy()
                preds.append(pred)
            
            ens_pred = np.stack(preds).mean(0)
                
            data_tpl[self.pred_key] = ens_pred

        # inside the postprocessing the result will be attached to the data_tpl
        self.postprocessing.postprocess_data_tpl(data_tpl, self.pred_key)

        torch.cuda.empty_cache()
        return data_tpl[self.pred_key]

    def save_prediction(self, data_tpl, folder_name, filename=None):

        self.models[0].save_prediction(data_tpl, folder_name, filename)

    def plot_prediction(self, data_tpl, folder_name, filename=None, image_key='image'):

        self.models[0].plot_prediction(data_tpl, folder_name, filename, image_key)

    def compute_error_metrics(self, data_tpl):
        return self.models[0].compute_error_metrics(data_tpl)

    def _init_global_metrics(self):
        self.global_metrics_helper = {}
        self.global_metrics = {}
        for c in self.models[0].lb_classes:
            self.global_metrics_helper.update({s+str(c): 0 for s in ['overlap_',
                                                                     'gt_volume_',
                                                                     'pred_volume_']})
            self.global_metrics.update({'dice_'+str(c): -1,
                                        'recall_'+str(c): -1,
                                        'precision_'+str(c): -1})

    def _update_global_metrics(self, data_tpl):

        if 'label' not in data_tpl:
            return

        if self.models[0].preprocessing.is_preprocessed_data_tpl(data_tpl):
            raise NotImplementedError('Ensemble prediction only implemented '
                                      'for raw data not for preprocessed data.')

        label = data_tpl['label']
        if self.models[0].preprocessing.reduce_lb_to_single_class:
            label = (label > 0).astype(label.dtype)
        pred = data_tpl[self.pred_key]

        # volume of one voxel
        fac = np.prod(data_tpl['spacing'])
        for c in self.models[0].lb_classes:
            lb_c = (label == c).astype(float)
            pred_c = (pred == c).astype(float)
            ovlp = self.global_metrics_helper['overlap_'+str(c)] + np.sum(lb_c * pred_c) * fac
            gt_vol = self.global_metrics_helper['gt_volume_'+str(c)] + np.sum(lb_c) * fac
            pred_vol = self.global_metrics_helper['pred_volume_'+str(c)] + np.sum(pred_c) * fac
            # update global dice, recall and precision
            if gt_vol + pred_vol > 0:
                self.global_metrics['dice_'+str(c)] = 200 * ovlp / (gt_vol + pred_vol)
            else:
                self.global_metrics['dice_'+str(c)] = 100
            if gt_vol > 0:
                self.global_metrics['recall_'+str(c)] = 100 * ovlp / gt_vol
            else:
                self.global_metrics['recall_'+str(c)] = 100 if pred_vol == 0 else 0
            if pred_vol > 0:
                self.global_metrics['precision_'+str(c)] = 100 * ovlp / pred_vol
            else:
                self.global_metrics['precision_'+str(c)] = 100 if gt_vol == 0 else 0

            # now update global metrics helper
            self.global_metrics_helper['overlap_'+str(c)] = ovlp
            self.global_metrics_helper['gt_volume_'+str(c)] = gt_vol
            self.global_metrics_helper['pred_volume_'+str(c)] = pred_vol

    def clean(self):
        for model in self.models:
            model.clean()

    def fill_cross_validation(self):
        
        ds = raw_Dataset(join(environ['OV_DATA_BASE'], 'raw_data', self.data_name),
                         prev_stages=self.prev_stages if hasattr(self, 'prev_stages') else None)
        pred_folder = join(environ['OV_DATA_BASE'], 'predictions', self.data_name,
                           self.preprocessed_name, self.model_name, 'cross_validation')
        for i in tqdm(range(len(ds))):
            data_tpl = ds[i]
            filename = data_tpl['scan'] + '.nii.gz'
            if filename not in listdir(pred_folder):
                self.__call__(data_tpl)
                self.save_prediction(data_tpl, folder_name='cross_validation')
