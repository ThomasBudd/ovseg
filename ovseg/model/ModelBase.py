from os.path import join, exists, basename
from ovseg.utils import io, path_utils
import os
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


NO_NAME_FOUND_WARNING_PRINTED = False


class ModelBase(object):
    '''
    The model holds everything that determines a cnn model:
        - preprocessing
        - augmentation
        - network
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
                      'preprocessed path was given. Searching...\n')
                preprocessed_folders = os.listdir(join(self.ov_data_base,
                                                       'preprocessed',
                                                       self.data_name))
                if len(preprocessed_folders) == 1:
                    print('Only one folder of preprocessed data found. '
                          'It is assumed that is it the right one.\n')
                    self.preprocessed_name = preprocessed_folders[0]
                elif self.model_name is preprocessed_folders:
                    print('Found preprocessed folder of the same name as the '
                          'model. Assume this is the right one.\n')
                    self.preprocessed_name = self.model_name
                elif 'default' in preprocessed_folders:
                    print('Found default preprocessed folder. Assume this is '
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
            else:
                self.preprocessed_path = join(self.ov_data_base,
                                              'preprocessed',
                                              self.data_name,
                                              'default')

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
            print('Loading model parameters')
            self.model_parameters = io.load_pkl(self.path_to_params)
            self.parameters_match_saved_ones = True
        elif params_given and not params_found:
            # typical case when first creating the model
            print('Saving model parameters to model base path')
            self.save_model_parameters()
            self.parameters_match_saved_ones = True
        else:
            model_params_from_pkl = io.load_pkl(self.path_to_params)
            if self.model_parameters == model_params_from_pkl:
                print('Input model parameters match pickled ones')
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
            print('Found '+self.network_name+' weights. Loading...')
            try:
                self.network.load_state_dict(torch.load(path_to_weights))
            except RuntimeError:
                print('WARNING! Weights could not be loaded. Something seems to be missmatching.')
        else:
            print('Found no preivous existing '+self.network_name+' weights. '
                  'Using random initialisation.')
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

    def predict(self, data_dict, mode='test'):
        raise NotImplementedError('predict function must be implemented in '
                                  'childclass.')

    def save_prediction(self, pred, data_dict, pred_folder, name):
        raise NotImplementedError('save_prediction not implemented')

    def plot_prediction(self, pred, data_dict, plot_folder, name):
        raise NotImplementedError('plot_prediction not implemented')

    def compute_error_metrics(self, pred, data_dict):
        raise NotImplementedError('compute_error_metrics not implemented')

    def _save_results_to_pkl_and_txt(self, results, path_to_store,
                                     name='validation'):

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

    def validate(self, save_preds=True, plot=True):
        '''
        iterates over the validation set and does the evaluation of the
        full 3d volumes. Results will be saved in the model (cv) path.
        '''
        global NO_NAME_FOUND_WARNING_PRINTED

        # we're going to store the validation results here
        val_folder = join(self.model_path, 'validation')
        folder_list = [val_folder]
        if save_preds:
            pred_folder = join(val_folder, 'predictions')
            folder_list.append(pred_folder)
        if plot:
            plot_folder = join(val_folder, 'plots')
            folder_list.append(plot_folder)

        # make folders
        for folder in folder_list:
            if not exists(folder):
                os.mkdir(folder)

        results = {}
        for i in tqdm(range(len(self.data.val_ds))):
            data_dict = self.data.val_ds[i]
            # first let's try to find the name
            if 'name' in data_dict.keys():
                name = data_dict['name']
            elif 'case' in data_dict.keys():
                name = data_dict['case']
            elif 'scan' in data_dict.keys():
                name = data_dict['scan']
            elif hasattr(self.data.val_ds, 'names'):
                name = basename(self.data.val_ds.names[i]).split('.')[0]
            elif hasattr(self.data.val_ds, 'scans'):
                name = basename(self.data.val_ds.scans[i]).split('.')[0]
            elif hasattr(self.data.val_ds, 'cases'):
                name = basename(self.data.val_ds.cases[i]).split('.')[0]
            else:
                d = str(int(np.ceil(np.log10(len(self.data.val_ds)))))
                name = 'case_%0'+d+'d'
                name = name % i
                if not NO_NAME_FOUND_WARNING_PRINTED:
                    print('Warning! Could not find a name for the prediction.'
                          'Please make sure that either the data of val_ds '
                          'has a key \'name\', \'case\' or \'scan\', or '
                          'that at least the data set has \'names\', \'scans\''
                          ' or \'cases\' as an attribute. Now we have to '
                          'choose the generic naming case_xxx')
                    NO_NAME_FOUND_WARNING_PRINTED = True

            # predict from this datapoint
            pred = self.predict(data_dict)

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
        self._save_results_to_pkl_and_txt(results, val_folder)

        # we also store the results in the CV folder and merge them with
        # possible other results from other folds
        if exists(join(self.model_cv_path, 'results.pkl')):
            print('Found exsiting results of other folds. Merge and save!')
            merged_results = io.load_pkl(join(self.model_cv_path,
                                              'results.pkl'))
            merged_results.update(results)
        else:
            print('Found no existing results of other folds. Saving only '
                  'these results.')
            merged_results = results

        # the merged results are kept in the model_cv_path
        self._save_results_to_pkl_and_txt(merged_results, self.model_cv_path)
