import numpy as np
from os.path import basename, join, exists


class SegmentationDataset(object):

    def __init__(self, scans, preprocessed_path):
        '''
        scans - list of scans to all volumes contained in this Dataset
        preprocessed_path - path to the folder where the prerprocessed data
                            is
        '''
        self.scans = scans
        self.preprocessed_path = preprocessed_path

        # these will carry all the pathes to data we need for training
        self.pathes_to_train_tuples = []
        for scan in self.scans:
            tple = [join(self.preprocessed_path, folder, scan)
                    for folder in ['images', 'labels']]
            self.pathes_to_train_tuples.append(tple)
        self.keys = ['image', 'label', 'orig_shape', 'orig_spacing']
        self.folders = ['images', 'labels', 'orig_shapes', 'spacings']
        for folder in self.folders:
            if not exists(join(self.preprocessed_path, folder)):
                raise FileNotFoundError('The preprocessed path to the '
                                        'segmentation data must have the '
                                        'folders ' + str(self.folders) + '. '
                                        + folder + ' was not found.')

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, ind=None):

        if ind is None:
            ind = np.random.randint(len(self.scans))
        else:
            ind = ind % len(self.scans)

        scan = self.scans[ind]
        data = {}
        for key, folder in zip(self.keys, self.folders):
            item = np.load(join(self.preprocessed_path, folder, scan))
            data[key] = item

        case = basename(scan).split('.')[0]
        data['case'] = case

        return data
