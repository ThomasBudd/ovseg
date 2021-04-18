import numpy as np
from os.path import basename, join, exists, isdir, split
from os import listdir
from ovseg.utils.io import read_data_tpl_from_nii, read_dcms


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
        data_dict['scan'] = name
        for key in ['dataset', 'pat_id', 'timepoint']:
            if key in f:
                name = name + '_' + f[key]
        data_dict['name'] = name

        return data_dict


class raw_Dataset(object):

    def __init__(self, raw_path, scans=None, image_folder=None, dcm_revers=True,
                 dcm_names_dict=None):

        assert image_folder in ['images', 'imagesTr', 'imagesTs', None]

        self.raw_path = raw_path
        all_im_folders = [imf for imf in listdir(self.raw_path) if imf.startswith('images')]
        all_lb_folders = [lbf for lbf in listdir(self.raw_path) if lbf.startswith('labels')]

        self.is_nifti = len(all_im_folders) > 0

        if self.is_nifti:

            if len(all_im_folders) > 1 and scans is None and image_folder is None:
                raise ValueError('Multiple image folders found at {}, but no scans were given '
                                 'neither was image_folder set. If there is more than one folder '
                                 'from [\'images\', \'imagesTr\', \'imagesTs\'] contained '
                                 'please specifiy which to read from or give a list of scans as '
                                 'input to raw_Dataset.')
            elif image_folder is not None:
                assert image_folder in all_im_folders
                self.image_folder = image_folder
            elif len(all_im_folders) == 1 and image_folder is None:
                self.image_folder = all_im_folders[0]

            # now the self.image_folder should be set

            if scans is None:
                # now try to get the scans
                labelfolder = 'labels' + self.image_folder[6:]
                if labelfolder in all_lb_folders:
                    self.scans = [scan[:-7] for scan in listdir(join(self.raw_path,
                                                                     labelfolder))]
                else:
                    self.scans = [scan[:-7] for scan in listdir(join(self.raw_path,
                                                                     self.image_folder))]
                    # check if we have medical decathlon style data
                    end_with_0000 = [scan for scan in scans if scan.endswith('_0000')]
                    if len(end_with_0000) > 0:
                        print('Found medical decathlon style data at '
                              + join(self.raw_path, self.image_folder))
                        self.scans = np.unique([scan[:-5] for scan in self.scans]).tolist()
            else:
                self.scans = scans

        else:
            # dcm case
            print('The folder {} was not identified as a nifti folder, assuming dcms are '
                  'contained.'.format(self.raw_path))
            self.dcm_revers = dcm_revers
            self.dcm_names_dict = dcm_names_dict
            if scans is None:
                self.scans = []
                folders = [join(self.raw_path, f) for f in listdir(self.raw_path)
                           if isdir(join(self.raw_path, f))]
                for folder in folders:
                    subfolders = [join(folder, f) for f in listdir(folder)
                                  if isdir(join(folder, f))]
                    if len(subfolders) > 0:
                        # if there are subfolders contained we will assume that these are the
                        # dcm folders we're looking for
                        self.scans.extend(subfolders)
                    elif len(listdir(folder)) > 0:
                        # otherwise if the folder is not empty we will assume that the dcms are here
                        self.scans.append(folder)
            else:
                self.scans = [join(self.raw_path, scan) for scan in scans]

        print('Using scans: ', [basename(scan) for scan in self.scans])

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, ind=None):

        if ind is None:
            ind = np.random.randint(len(self.scans))
        else:
            ind = ind % len(self.scans)

        scan = self.scans[ind]

        if self.is_nifti:
            data_tpl = read_data_tpl_from_nii(self.raw_path, scan)
        else:
            data_tpl = read_dcms(join(self.raw_path, scan),
                                 reverse=self.dcm_revers,
                                 names_dict=self.dcm_names_dict,
                                 dataset=basename(self.raw_path))
            path, folder = split(scan)
            if basename(path) == self.raw_path:
                scan = folder
            else:
                path, superfolder = split(path)
                if 'pat_name' in data_tpl and 'date' in data_tpl:
                    scan = data_tpl['pat_name'] + '_' + data_tpl['date']
                else:
                    scan = superfolder + '_' + folder

        data_tpl['scan'] = scan

        return data_tpl
