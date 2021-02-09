from os.path import join, exists, basename
from ovseg.utils import io, path_utils
from ovseg.utils.dict_equal import dict_equal
import os
import torch
from tqdm import tqdm
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

    def __init__(self, val_fold: int, data_name: str, model_name: str,
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

        # the model path will be pointing to the model of this particular
        # fold
        self.ov_data_base = os.environ['OV_DATA_BASE']
        self.model_cv_path = join(self.ov_data_base,
                                  'trained_models',
                                  self.data_name,
                                  self.model_name)
        self.model_path = join(self.model_cv_path, 'fold_%d' % self.val_fold)
        path_utils.maybe_create_path(self.model_path)
        self.path_to_params = join(self.model_cv_path, self.model_parameters_name+'.pkl')
        if self.preprocessed_name is None:
            if not self.is_inference_only:
                print('Model was called not in inference mode and no '
                      'preprocessed path was given. Searching for preprocessed data...\n')
                preprocessed_folders = os.listdir(join(self.ov_data_base,
                                                       'preprocessed',
                                                       self.data_name))
                if len(preprocessed_folders) == 1:
                    self.preprocessed_name = preprocessed_folders[0]
                    print('Only one folder of preprocessed data found ({}). '
                          'It is assumed that is it the right one.\n'
                          ''.format(self.preprocessed_name))
                elif self.model_name is preprocessed_folders:
                    print('Found preprocessed folder of the same name as the '
                          'model. Assume this is the right one.\n')
                    self.preprocessed_name = self.model_name
                elif 'default' in preprocessed_folders:
                    print('Found default preprocessed folder (default). Assume this is '
                          'the right one.\n')
                    self.preprocessed_name = 'default'
                else:
                    raise ValueError('No name for preprocessed data was given,'
                                     ' even though the model was not '
                                     'initialised in inference mode. If you '
                                     'want to train you have to know where the'
                                     ' data is.')

                # if we've made it until here there was no error raised.
                self.preprocessed_path = join(self.ov_data_base,
                                              'preprocessed',
                                              self.data_name,
                                              self.preprocessed_name)
            # else:
            # in inference mode we shouldn't have a preprocessed path
            #     self.preprocessed_path = join(self.ov_data_base,
            #                                   'preprocessed',
            #                                   self.data_name,
            #                                   'default')

        else:
            self.preprocessed_path = join(self.ov_data_base,
                                          'preprocessed',
                                          self.data_name,
                                          self.preprocessed_name)

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
                print('Found conflict between saved and inputed model parameters. '
                      'New paramters added will not be stored in the .pkl file automatically. '
                      'If you want to overwrite, call model.save_model_parameters(). '
                      'Make sure you want to alter the parameters stored at '+self.path_to_params)

                self.parameters_match_saved_ones = False

        # %% now initialise everything we need
        self.initialise_preprocessing()
        self.initialise_augmentation()
        self.initialise_network()
        path_to_weights = join(self.model_path, self.network_name + '_weights')
        if exists(path_to_weights):
            print('Found '+self.network_name+' weights. Loading...\n\n')
            try:
                self.network.load_state_dict(torch.load(path_to_weights))
                print('Done!\n')
            except RuntimeError:
                print('WARNING! Weights could not be loaded. Something seems to be missmatching.')
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

    def _write_parameter_dict_to_txt(self, dict_name, param_dict, file,
                                     n_tabs):
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

    def predict(self, data_dict, is_preprocessed):
        raise NotImplementedError('predict function must be implemented in '
                                  'childclass.')

    def save_prediction(self, pred, data_dict, pred_folder, name):
        raise NotImplementedError('save_prediction not implemented')

    def plot_prediction(self, pred, data_dict, plot_folder, name):
        raise NotImplementedError('plot_prediction not implemented')

    def compute_error_metrics(self, pred, data_dict):
        raise NotImplementedError('compute_error_metrics not implemented')

    def _save_results_to_pkl_and_txt(self, results, path_to_store, name):

        file_name = name + '_results'

        # thats the easy part!
        io.save_pkl(results, join(path_to_store, file_name+'.pkl'))

        # now writing everything to a txt file we can nicely look at
        # yeah this might be stupid but it helped me couple of times in the
        # past...
        cases = sorted(list(results.keys()))
        if len(cases) == 0:
            print('results dict is empty.')
            return

        metric_names = list(results[cases[0]].keys())
        metrics = np.array([[results[case][metric] for metric in metric_names] for case in cases])
        means = np.nanmean(metrics, 0)
        medians = np.nanmedian(metrics, 0)
        with open(join(path_to_store, file_name+'.txt'), 'w') as file:
            file.write('RESULTS:\n')
            file.write('\n')
            for i, metric in enumerate(metric_names):
                file.write(metric + ':\n')
                s = '\t Mean: '+self.fmt_write+', Median: '+self.fmt_write+'\n'
                file.write(s.format(means[i], medians[i]))
            file.write('\n')
            file.write('\n')
            for j, case in enumerate(cases):
                file.write(case + ':\n')
                s = '\t ' + ', '.join([metric+': '+self.fmt_write
                                       for metric in metric_names])
                file.write(s.format(*metrics[j]) + '\n')

    def eval_ds(self, ds, is_preprocessed, ds_name, save_preds=True, plot=True,
                force_evaluation=False):
        '''
        iterates over the validation set and does the evaluation of the
        full 3d volumes. Results will be saved in the model (cv) path.
        '''
        global NO_NAME_FOUND_WARNING_PRINTED

        # we're going to store the validation results here
        eval_folder = join(self.model_path, ds_name)
        folder_list = [eval_folder]
        if save_preds:
            pred_folder = join(eval_folder, 'predictions')
            folder_list.append(pred_folder)
        if plot:
            plot_folder = join(eval_folder, 'plots')
            folder_list.append(plot_folder)

        if not force_evaluation:
            exs_folders = [exists(folder) for folder in folder_list]
            exs_files = [exists(join(self.model_path, ds_name+'_results.'+ext)) for ext in
                         ['txt', 'pkl']]
            if np.all(exs_folders) and np.all(exs_files):
                print('Found existing evaluation folders and files of dataset ' + ds_name +
                      '. Their content wasn\'t checked, but the evaluation will be skipped.\n'
                      'If you want to force the evaluation please delete the old files and folders '
                      'or pass force_evaluation=True.\n\n')
                return

        # make folders
        for folder in folder_list:
            if not exists(folder):
                os.mkdir(folder)

        results = {}
        print('Evaluation '+ds_name+'...\n\n')
        for i in tqdm(range(len(ds))):
            data_dict = ds[i]
            # first let's try to find the name
            if 'name' in data_dict.keys():
                name = data_dict['name']
            else:
                d = str(int(np.ceil(np.log10(len(ds)))))
                name = 'case_%0'+d+'d'
                name = name % i
                if not NO_NAME_FOUND_WARNING_PRINTED:
                    print('Warning! Could not find a name for the prediction.'
                          'Please make sure that the items of the dataset have a key \'name\'.'
                          'Choose generic naming case_xxx as names.\n')
                    NO_NAME_FOUND_WARNING_PRINTED = True

            # predict from this datapoint
            pred = self.predict(data_dict, is_preprocessed)
            if torch.is_tensor(pred):
                pred = pred.cpu().numpy()

            # now compute the error metrics that we like
            metrics = self.compute_error_metrics(pred, data_dict)
            results[name] = metrics

            # store the prediction for example as nii files
            if save_preds:
                self.save_prediction(pred, data_dict, pred_folder, name)

            # plot the results, maybe just a single slice?
            if plot:
                self.plot_prediction(pred, data_dict, plot_folder, name)

        # iteration done. Let's store the results and get out of here!
        # first we store the results for this fold in the validation folder
        self._save_results_to_pkl_and_txt(results, self.model_path, name=ds_name)

        # we also store the results in the CV folder and merge them with
        # possible other results from other folds
        path_to_results = join(self.model_cv_path, ds_name+'_CV_results.pkl')
        # to differentiate by name what comes from which fold we add fold_x to the names
        results_fold = {key+'_fold_{}'.format(self.val_fold): results[key] for key in results}
        if exists(path_to_results):
            print('Found exsiting results of other folds in CV path. Merge and save!\n')
            merged_results = io.load_pkl(path_to_results)
            merged_results.update(results_fold)
        else:
            print('Found no existing results of other folds in CV path. Saving only '
                  'these results.\n')
            merged_results = results_fold

        # the merged results are kept in the model_cv_path
        self._save_results_to_pkl_and_txt(merged_results, self.model_cv_path, name=ds_name+'_CV')

    def eval_validation_set(self, save_preds=True, plot=True, force_evaluation=False):
        self.eval_ds(self.data.val_ds, is_preprocessed=True, ds_name='validation',
                     save_preds=save_preds, plot=plot, force_evaluation=force_evaluation)

    def eval_training_set(self, save_preds=False, plot=True, force_evaluation=False):
        self.eval_ds(self.data.trn_ds, is_preprocessed=True, ds_name='training',
                     save_preds=save_preds, plot=plot, force_evaluation=force_evaluation)
