from ovseg.data.DataBase import DataBase
from ovseg.data.ImageProcessingDataloader import ImageProcessingDataloader
from os import listdir
from os.path import join


class ImageProcessingData(DataBase):

    def initialise_dataloader(self, is_train):
        if is_train:
            print('Initialise training dataloader')
            self.trn_dl = ImageProcessingDataloader(self.trn_ds,
                                                    **self.trn_dl_params)
        else:
            print('Initialise validation dataloader')
            self.val_dl = ImageProcessingDataloader(self.val_ds,
                                                    **self.val_dl_params)
