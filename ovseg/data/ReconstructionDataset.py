import numpy as np
from os.path import join, exists, basename


class ReconstructionDataset(object):

    def __init__(self, scans, preprocessed_path, dose='full'):
        self.scans = scans
        self.preprocessed_path = preprocessed_path
        self.dose = dose

        # now make the pathes to the tuples we're interested in
        self.pathes_to_train_tuples = []
        for scan in self.scans:
            tple = [join(self.preprocessed_path, folder, scan)
                    for folder in ['projections_'+self.dose, 'images']]
            self.pathes_to_train_tuples.append(tple)
        self.keys = ['proj', 'image', 'spacing']
        self.folders = ['projections_'+self.dose, 'images', 'spacings']
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
