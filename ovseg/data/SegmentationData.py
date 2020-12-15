from ovseg.data.DataBase import DataBase
from ovseg.data.SegmentationDataset import SegmentationDataset
from ovseg.data.SegmentationDataloader import SegmentationDataloader
from os import listdir
from os.path import join


class SegmentationData(DataBase):

    def initialise_dataloader(self, is_train):
        if is_train:
            print('Initialise training dataloader')
            self.trn_dl = SegmentationDataloader(self.trn_ds,
                                                 **self.trn_dl_params)
        else:
            print('Initialise validation dataloader')
            try:
                self.val_dl = SegmentationDataloader(self.val_ds,
                                                     **self.val_dl_params)
            except (AttributeError, TypeError):
                print('No validatation dataloader initialised')
                self.val_dl = None
