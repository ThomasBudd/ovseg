import numpy as np
from os.path import basename, join, exists


class Dataset(object):

    def __init__(self, scans, preprocessed_path, keys, folders, **kwargs):
        '''
        scans - list of scans to all volumes contained in this Dataset
        preprocessed_path - path to the folder where the prerprocessed data
                            is
        '''
        self.scans = scans
        self.preprocessed_path = preprocessed_path
        self.keys = keys
        self.folders = folders

        for folder in self.folders:
            if not exists(join(self.preprocessed_path, folder)):
                raise FileNotFoundError('The preprocessed path to the '
                                        'data must have the '
                                        'folders ' + str(self.folders) + '. '
                                        + folder + ' was not found.')

        # these will carry all the pathes to data we need for training
        self.path_dicts = []
        for scan in self.scans:
            path_dict = {key: join(self.preprocessed_path, folder, scan)
                         for key, folder in zip(self.keys, self.folders)}
            if np.all([exists(path_dict[key]) for key in self.keys]):
                self.path_dicts.append(path_dict)
            else:
                print('Warning some .npy files of scan {} missing'.format(scan))

        for key in kwargs:
            print('Got unexcpected keyword '+key+' with value' +
                  str(kwargs[key]) + ' as input to dataset.')

    def __len__(self):
        return len(self.path_dicts)

    def __getitem__(self, ind=None):

        if ind is None:
            ind = np.random.randint(len(self.scans))
        else:
            ind = ind % len(self.scans)

        path_dict = self.path_dicts[ind]
        data_dict = {key: np.load(path_dict[key]) for key in self.keys}

        # last but not least the name and fingerprint
        scan = basename(path_dict[self.keys[0]])
        path_to_fp = join(self.preprocessed_path, 'fingerprints', scan)
        if exists(path_to_fp):
            f = np.load(path_to_fp, allow_pickle=True).item()
            data_dict.update(f)
        name = basename(scan).split('.')[0]
        for key in ['dataset', 'pat_id', 'timepoint']:
            if key in f:
                name = name + '_' + f[key]
        data_dict['name'] = name

        return data_dict
