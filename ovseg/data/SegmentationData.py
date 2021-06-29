from ovseg.data.DataBase import DataBase
from ovseg.data.SegmentationDataloader import SegmentationDataloader
from os import listdir
from os.path import join


class SegmentationData(DataBase):

    def __init__(self, augmentation=None, *args, **kwargs):
        self.augmentation = augmentation
        super().__init__(*args, **kwargs)

    def initialise_dataloader(self, is_train):
        if is_train:
            print('Initialise training dataloader')
            self.trn_dl = SegmentationDataloader(self.trn_ds,
                                                 augmentation=self.augmentation,
                                                 **self.trn_dl_params)
        else:
            print('Initialise validation dataloader')
            try:
                self.val_dl = SegmentationDataloader(self.val_ds,
                                                     augmentation=self.augmentation,
                                                     **self.val_dl_params)
            except (AttributeError, TypeError):
                print('No validatation dataloader initialised')
                self.val_dl = None

    def clean(self):
        self.trn_dl.dataset._maybe_clean_stored_data()
        if self.val_dl is not None:
            self.val_dl.dataset._maybe_clean_stored_data()
