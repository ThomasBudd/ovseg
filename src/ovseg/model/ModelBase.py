from os.path import join, exists
from ovseg.utils import io, path_utils
from ovseg.utils.dict_equal import dict_equal, print_dict_diff
from ovseg.data.Dataset import raw_Dataset
from ovseg import OV_PREPROCESSED
import os
import torch
from time import sleep, asctime
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    print('No tqdm found, using no pretty progressing bars')
    tqdm = lambda x: x
import numpy as np


NO_NAME_FOUND_WARNING_PRINTED = False


class ModelBase(object):
    '''
    The model holds everything that determines a cnn model:
        - preprocessing
        - augmentation
        - network
        - prediction
        - postprocessing
        - data
        - training

    if is_inference_only the data and training will not be initialised.
    The use of this class is to wrapp up all this information, to train the
    network and to evaluate it on a full 3d volume.
    '''

    def __init__(self, val_fold, data_name: str, model_name: str,
                 model_parameters=None, preprocessed_name=None,
                 network_name='network', is_inference_only: bool = False,
                 fmt_write='{:.4f}', model_parameters_name='model_parameters'):
        # keep all the args
        self.val_fold = val_fold
        self.data_name = data_name
        self.model_name = model_name
        self.preprocessed_name = preprocessed_name
        self.model_parameters = model_parameters
        self.network_name = network_name
        self.is_inference_only = is_inference_only
        self.fmt_write = fmt_write
        self.model_parameters_name = model_parameters_name
        self.ov_data_base = os.environ['OV_DATA_BASE']

        if isinstance(self.val_fold, int):
            self.val_fold_str = 'fold_' + str(self.val_fold)
        else:
            assert isinstance(self.val_fold, (tuple, list)), "val_fold must be int, list or tuple"
            self.val_fold_str = 'ensemble_'+'_'.join([str(f) for f in self.val_fold])

        # just to enable CPU execution where the GPU is missing
        self.dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        if self.preprocessed_name is None:
            path_to_preprocessed_data = join(OV_PREPROCESSED,
                                             self.data_name)
            if not exists(path_to_preprocessed_data):
                raise FileNotFoundError('Path to preprocessed data doesn\'t exsist. Make sure '
                                        'to preprocess your raw data before using models.')

            preprocessed_folders = os.listdir(path_to_preprocessed_data)

            if not len(preprocessed_folders) == 1:
                raise FileNotFoundError('No input \'preprocessed_name\' was given and it could '
                                        'not be identified automatically. Available preprocessed '
                                        'data folders are {}'.format(preprocessed_folders))
            else:
                print('No preprocessed_name given, chose {}.'.format(preprocessed_folders[0]))
                self.preprocessed_name = preprocessed_folders[0]

        # set the path to the preprocessed data
        self.preprocessed_path = join(OV_PREPROCESSED,
                                      self.data_name,
                                      self.preprocessed_name)

        # the model path will be pointing to the model of this particular fold
        # weights and (hyper) parameters are stored here
        self.model_cv_path = join(self.ov_data_base,
                                  'trained_models',
                                  self.data_name,
                                  self.preprocessed_name,
                                  self.model_name)
        self.model_path = join(self.model_cv_path, self.val_fold_str)
        path_utils.maybe_create_path(self.model_path)
        self.path_to_params = join(self.model_cv_path, self.model_parameters_name+'.pkl')

        # %% check and load model_parameters
        params_given = isinstance(self.model_parameters, dict)
        params_found = exists(self.path_to_params)

        if not params_found and not params_given:
            # we need either as input
            raise FileNotFoundError('The model parameters were neither given '
                                    'as input, nor found at ' +
                                    self.model_cv_path+'.')
        elif not params_given and params_found:
            # typical case when loading the model
            print('Loading model parameters.\n')
            self.model_parameters = io.load_pkl(self.path_to_params)
            self.parameters_match_saved_ones = True
        elif params_given and not params_found:
            # typical case when first creating the model
            print('Saving model parameters to model base path.\n')
            self.save_model_parameters()
            self.parameters_match_saved_ones = True
        else:
            model_params_from_pkl = io.load_pkl(self.path_to_params)
            if dict_equal(self.model_parameters, model_params_from_pkl):
                print('Input model parameters match pickled ones.\n')
                self.parameters_match_saved_ones = True
            else:
                print('-------Found conflict between saved and inputed model parameters-------')
                print_dict_diff(self.model_parameters, model_params_from_pkl, 'input paramters'
                                'pkl paramters')
                print('-----------------------------------------------------------------------')
                print('The inputed paramters will are NOT overwriting the pkl parameter. \n '
                      'If you want to overwrite, call model.save_model_parameters(). '
                      'Make sure you want to alter the parameters stored at '+self.path_to_params)

                self.parameters_match_saved_ones = False

        # %% now initialise everything we need

        # this is what we once did, not let's just identify the prediction by the ugly long key...
        self.pred_key = '_'.join(['prediction', self.data_name, self.preprocessed_name,
                                  self.model_name])
        # if 'prediction_key' in self.model_parameters:
        #     self.pred_key = self.model_parameters['prediction_key']
        # else:
        #     print('\'prediction_key\' was not found in model_parameters.'
        #           'Init as \'prediction\'.')
        #     self.pred_key = 'prediction'
        #     self.model_parameters['prediction_key'] = 'prediction'
        #     if self.parameters_match_saved_ones:
        #         self.save_model_parameters()

        self.initialise_preprocessing()
        self.initialise_augmentation()
        self.initialise_network()
        path_to_weights = join(self.model_path, self.network_name + '_weights')
        if exists(path_to_weights):
            print('Found '+self.network_name+' weights. Loading from '+path_to_weights+'\n\n')
            self.network.load_state_dict(torch.load(path_to_weights,
                                         map_location=torch.device(self.dev)))
        else:
            print('Found no preivous existing '+self.network_name+' weights. '
                  'Using random initialisation.\n')
        self.initialise_postprocessing()

        if not self.is_inference_only:
            if self.preprocessed_name is None:
                raise ValueError('The \'preprocessed_name\' must be given when'
                                 ' the model is initialised for training. '
                                 'preprocessed data is expected to be in '
                                 'OV_DATA_BASE/preprocessed/data_folder'
                                 '/preprocessed_name')
            self.initialise_data()
            self.initialise_training()
        self._model_parameters_to_txt()

    # %% this is just putting the parameters in a nice .txt file so that
    # we can easily see our choices
    def _model_parameters_to_txt(self):
        file_name = self.model_parameters_name+'.txt'

        with open(join(self.model_cv_path, file_name), 'w') as file:
            self._write_parameter_dict_to_txt(self.model_parameters_name,
                                              self.model_parameters, file, 0)

    def _write_parameter_dict_to_txt(self, dict_name, param_dict, file, n_tabs):
        # recurively go down all dicts and print their content
        # each time we go a dict deeper we add another tab for more beautiful
        # nested printing
        tabs = ''.join(n_tabs * ['\t'])
        s = tabs + dict_name + ' =\n'
        file.write(s)
        for key in param_dict.keys():
            item = param_dict[key]
            if isinstance(item, dict):
                self._write_parameter_dict_to_txt(key, item, file, n_tabs+1)
            else:
                s = tabs + '\t' + key + ' = ' + str(item) + '\n'
                file.write(s)

    def save_model_parameters(self):
        io.save_pkl(self.model_parameters, self.path_to_params)
        self._model_parameters_to_txt()

    # %% functions every childclass should have
    def initialise_preprocessing(self):
        raise NotImplementedError('initialise_preprocessing not implemented. '
                                  'If your model needs no preprocessing please'
                                  ' implement an empty function.')

    def initialise_augmentation(self):
        raise NotImplementedError('initialise_augmentation not implemented. '
                                  'If your model needs no augmenation please'
                                  ' implement an empty function.')

    def initialise_network(self):
        raise NotImplementedError('initialise_network not implemented.')

    def initialise_postprocessing(self):
        raise NotImplementedError('initialise_postprocessing not implemented. '
                                  'If your model needs no postprocessing '
                                  'please implement an empty function.')

    def initialise_data(self):
        raise NotImplementedError('initialise_data not implemented. '
                                  'If your model needs no data please'
                                  ' implement an empty function or if you '
                                  'don\'t need training use '
                                  '\'is_inferece_only=True.')

    def initialise_training(self):
        raise NotImplementedError('initialise_training not implemented. '
                                  'If your model needs no training please'
                                  ' implement an empty function, or use '
                                  '\'is_inferece_only=True.')

    def predict(self, data_tpl):
        raise NotImplementedError('predict function must be implemented in '
                                  'childclass.')

    def save_prediction(self, data_tpl, ds_name, filename=None):
        raise NotImplementedError('save_prediction not implemented')

    def plot_prediction(self, data_tpl, ds_name, filename=None):
        raise NotImplementedError('plot_prediction not implemented')

    def compute_error_metrics(self, data_tpl):
        raise NotImplementedError('compute_error_metrics not implemented')

    def _init_global_metrics(self):
        self.global_metrics = {}
        print('computing no global error metrics')

    def _update_global_metrics(self, data_tpl):
        return None

    def _save_results_to_pkl_and_txt(self, results, path_to_store, ds_name, names_for_txt=None):

        file_name = ds_name + '_results'

        if names_for_txt is None:
            names_for_txt = {k: k for k in results.keys()}

        # thats the easy part!
        io.save_pkl(results, join(path_to_store, file_name+'_per_scan.pkl'))
        
        if hasattr(self, 'global_metrics'):
            if len(self.global_metrics) > 0:
                io.save_pkl(self.global_metrics, join(path_to_store, file_name+'_global.pkl'))

        # now writing everything to a txt file we can nicely look at
        # yeah this might be stupid but it helped me couple of times in the
        # past...
        cases = sorted(list(results.keys()))
        if len(cases) == 0:
            print('results dict is empty.')
            return None

        metric_names = list(results[cases[0]].keys())
        metrics = np.array([[results[case][metric] for metric in metric_names] for case in cases])
        means = np.nanmean(metrics, 0)
        medians = np.nanmedian(metrics, 0)
        with open(join(path_to_store, file_name+'.txt'), 'w') as file:
            # first we write the global stats
            file.write(asctime() + '\n')
            if hasattr(self, 'global_metrics'):
                file.write('GLOBAL RESULTS:\n')
                file.write('\n')
                for metric in self.global_metrics:
                    if not metric.startswith('_'):
                        s = metric+': '+self.fmt_write+'\n'
                        file.write(s.format(self.global_metrics[metric]))

            # now the per volume results from the results dict
            # here the mean and median
            file.write('PER VOLUME RESULTS:\n')
            file.write('\n')
            for i, metric in enumerate(metric_names):
                file.write(metric + ':\n')
                s = '\t Mean: '+self.fmt_write+', Median: '+self.fmt_write+'\n'
                file.write(s.format(means[i], medians[i]))
            file.write('\n')
            file.write('\n')

            # now the results for each case
            for j, case in enumerate(cases):
                file.write(names_for_txt[case] + ':\n')
                s = '\t ' + ', '.join([metric+': '+self.fmt_write
                                       for metric in metric_names])
                file.write(s.format(*metrics[j]) + '\n')

    def eval_ds(self, ds, ds_name: str, save_preds: bool = True, save_plots: bool = False,
                force_evaluation: bool = False, merge_to_CV_results: bool = False,
                save_folder_name=None):
        '''

        Parameters
        ----------
        ds : Dataset
            Dataset type object that has a length and return a data_tpl for each index
        ds_name : string
            name of the dataset used when saving the results
        save_preds : bool, optional
            if save_preds the predictions are kept in the "predictions" folder at OV_DATA_BASE.
            The default is True.
        save_plots : bool, optional
            if save_preds the predictions are kept in the "predictions" folder at OV_DATA_BASE.
        force_evaluation : bool, optional
            if not force_evaluation the results files and folders are superficially checked.
            If everything seems to be there we skip the evaluation
            The default is False.
        merge_to_CV_results : bool, optional
            Set true only for the validation set.
            Results are merged with the ones from other folds and stored in the CV path.
            The default is False.

        Returns
        -------
        None.

        '''

        if len(ds) == 0:
            print('Got empty dataset for evaluation. Nothing to do here --> leaving!')
            return

        global NO_NAME_FOUND_WARNING_PRINTED

        if save_folder_name is None:
            save_folder_name = ds_name + '_' + self.val_fold_str

        # first check if the evaluation is already done and quit in case we don't want to force
        # the evaluation
        if not force_evaluation:
            # first check if the two results file exist
            # if the files do not exist we have an indicator that we have to repeat
            # the evaluation
            do_evaluation = not np.all([exists(join(self.model_path, ds_name+'_results_per_scan.'+ext))
                                        for ext in ['txt', 'pkl']])

            if save_preds:
                # next check if the prediction folder exists
                pred_folder = os.path.join(os.environ['OV_DATA_BASE'], 'predictions',
                                           self.data_name,
                                           self.preprocessed_name,
                                           self.model_name,
                                           save_folder_name)
                if not exists(pred_folder):
                    do_evaluation = True

            if save_plots:
                # same for the plot folder, if it doesn't exsist we do the prediction
                plot_folder = os.path.join(os.environ['OV_DATA_BASE'], 'plots',
                                           self.data_name,
                                           self.preprocessed_name,
                                           self.model_name,
                                           save_folder_name)
                if not exists(plot_folder):
                    do_evaluation = True

            if not do_evaluation:
                print('Found existing evaluation folders and files for this dataset (' + ds_name +
                      '). Their content wasn\'t checked, but the evaluation will be skipped.\n'
                      'If you want to force the evaluation please delete the old files and folders '
                      'or pass force_evaluation=True.\n\n')
                if merge_to_CV_results:
                    print('Merging resuts to CV....')
                    self._merge_results_to_CV(ds_name)
                return

        self._init_global_metrics()
        results = {}
        names_for_txt = {}
        print('Evaluating '+ds_name+'...\n\n')
        sleep(1)
        for data_tpl in tqdm(ds):
            # first let's try to find the name
            scan = data_tpl['scan']

            if 'name' in data_tpl.keys():
                names_for_txt[scan] = data_tpl['name']
            else:
                names_for_txt[scan] = scan

            # predict from this datapoint
            pred = self.__call__(data_tpl)
            if torch.is_tensor(pred):
                pred = pred.cpu().numpy()

            # now compute the error metrics that we like
            metrics = self.compute_error_metrics(data_tpl)
            if metrics is not None:
                results[scan] = metrics
            self._update_global_metrics(data_tpl)

            # store the prediction for example as nii files
            if save_preds:
                self.save_prediction(data_tpl, folder_name=save_folder_name, filename=scan)

            # plot the results, maybe just a single slice?
            if save_plots:
                self.plot_prediction(data_tpl, folder_name=save_folder_name, filename=scan)

        # iteration done. Let's store the results and get out of here!
        # first we store the results for this fold in the validation folder
        self._save_results_to_pkl_and_txt(results, self.model_path, ds_name=ds_name)

        if merge_to_CV_results:
            print('Merging resuts to CV....')
            self._merge_results_to_CV(ds_name)

    def _merge_results_to_CV(self, ds_name='validation'):
        # we also store the results in the CV folder and merge them with
        # possible other results from other folds
        # to differentiate by name what comes from which fold we add fold_x to the names
        merged_results = {}
        all_folds = [fold for fold in os.listdir(self.model_cv_path) if fold.startswith('fold')]
        for fold in all_folds:
            if ds_name+'_results_per_scan.pkl' in os.listdir(os.path.join(self.model_cv_path, fold)):
                fold_results = io.load_pkl(os.path.join(self.model_cv_path, fold,
                                                        ds_name+'_results_per_scan.pkl'))
                merged_results.update({key+'_'+fold: fold_results[key] for key in fold_results})
        
        self._merge_global_results_to_CV(ds_name)
        
        # the merged results are kept in the model_cv_path
        self._save_results_to_pkl_and_txt(merged_results,
                                          self.model_cv_path,
                                          ds_name=ds_name+'_CV')

    def _merge_global_results_to_CV(self, ds_name):
        print('merging no global metrics')

    def eval_validation_set(self, save_preds=True, save_plots=False, force_evaluation=False):
        if not hasattr(self.data, 'val_ds'):
            print('No validation data found! Skipping prediction...')
            return

        self.eval_ds(self.data.val_ds, ds_name='validation',
                     save_preds=save_preds, save_plots=save_plots,
                     force_evaluation=force_evaluation,
                     merge_to_CV_results=True, save_folder_name='cross_validation')

    def eval_training_set(self, save_preds=False, save_plots=False, force_evaluation=False):
        self.eval_ds(self.data.trn_ds, ds_name='training',
                     save_preds=save_preds, save_plots=save_plots,
                     force_evaluation=force_evaluation)

    def eval_raw_dataset(self, data_name, save_preds=True, save_plots=False,
                         force_evaluation=False, scans=None, image_folder=None, dcm_revers=True,
                         dcm_names_dict=None):
        ds = raw_Dataset(join(os.environ['OV_DATA_BASE'], 'raw_data', data_name),
                         scans=scans,
                         image_folder=image_folder,
                         dcm_revers=dcm_revers,
                         dcm_names_dict=dcm_names_dict,
                         prev_stages=self.prev_stages if hasattr(self, 'prev_stages') else None)
        self.eval_ds(ds, ds_name=data_name, save_preds=save_preds, save_plots=save_plots,
                     force_evaluation=force_evaluation)
