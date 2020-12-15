import numpy as np
from os.path import exists, join, basename
from ovseg.utils import io
from os import environ, sep, listdir
import pickle
from ovseg.data.utils import split_scans_random_uniform, \
    split_scans_by_patient_id
from ovseg.data.Dataset import Dataset


class DataBase():

    def __init__(self, val_fold, preprocessed_path, keys, folders, n_folds=5,
                 fixed_shuffle=True, trn_dl_params={}, ds_params={},
                 val_dl_params={}):
        '''
        DataBase(val_fold, preprocessed_path, n_folds=5, fixed_shuffle=True,
                 trn_dl_params={}, ds_params={}, val_dl_params={})

        Basic class that splits data and creates train and validation datasets
        and dataloaders. Can be initialised gived fixed folds \'folds\' or
        with the scans to all datatuples. In the latter case this class does
        the splitting automatically.
        '''
        # set number of validation fold
        self.val_fold = val_fold
        self.preprocessed_path = preprocessed_path
        self.keys = keys
        self.folders = folders
        # other arguments. we don't need them when finding exisiting splits
        # but anyways...
        self.n_folds = n_folds
        self.fixed_shuffle = fixed_shuffle
        # now save the additional arguments for creating the datasets
        # and dataloaders
        self.ds_params = ds_params
        self.trn_dl_params = trn_dl_params
        self.val_dl_params = val_dl_params

        # let the important bit start: The splitting of the data
        # check if there is alreay some split
        path_to_splits = join(self.preprocessed_path, 'splits.pkl')
        if exists(path_to_splits):
            # in this case a split of data is given
            print('Found existing data split')
            self.splits = io.load_pkl(path_to_splits)
            self.n_folds = len(self.splits)
        else:
            print('No data split found.')
            print('Computing new one..')

            self.scans = listdir(join(self.preprocessed_path, self.folders[0]))
            # try to find existing split
            data_name = self.preprocessed_path.split(sep)[-2]
            ov_data_base = environ['OV_DATA_BASE']
            if exists(join(self.preprocessed_path, 'decode.pkl')):
                decode = io.load_pkl(join(self.preprocessed_path,
                                          'decode.pkl'))
                self.splits = split_scans_by_patient_id(self.scans, decode,
                                                        self.n_folds,
                                                        self.fixed_shuffle)
            elif exists(join(ov_data_base, data_name, 'decode.pkl')):
                decode = io.load_pkl(join(ov_data_base, data_name,
                                          'decode.pkl'))
                self.splits = split_scans_by_patient_id(self.scans, decode,
                                                        self.n_folds,
                                                        self.fixed_shuffle)
            else:
                print('WARNING: no file decode.pkl found in neither the '
                      'preprocessed nor the raw data path. Random uniform '
                      'splitting is applied, make sure to create the '
                      'decode.pkl file if patients have multiple scans in '
                      'your data.\n In this case remove the splits file and '
                      'leave the decode.pkl instead.')
                self.splits = split_scans_random_uniform(self.scans,
                                                         self.n_folds,
                                                         self.fixed_shuffle)
            io.save_pkl(self.splits, path_to_splits)
            print('New split saved')

        self.split = self.splits[self.val_fold]
        self.trn_scans = self.split['train']
        self.val_scans = self.split['val']

        # now create datasets
        self.initialise_dataset(is_train=True)
        self.initialise_dataset(is_train=False)

        # and the dataloaders
        self.initialise_dataloader(is_train=True)
        self.initialise_dataloader(is_train=False)

    def initialise_dataset(self, is_train):
        if is_train:
            self.trn_ds = Dataset(self.trn_scans, self.preprocessed_path,
                                  self.keys, self.folders, **self.ds_params)
        else:
            self.val_ds = Dataset(self.val_scans, self.preprocessed_path,
                                  self.keys, self.folders, **self.ds_params)

    def initialise_dataloader(self, is_train):
        raise NotImplementedError('function \'initialise_dataloader\' was not '
                                  ' overloaded in child class. This function '
                                  'need to create the attributes trn_dl and '
                                  'val_dl.')
