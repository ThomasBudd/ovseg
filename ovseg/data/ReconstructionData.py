from ovseg.data.DataBase import DataBase
from ovseg.data.ReconstructionDataset import ReconstructionDataset
from ovseg.data.ReconstructionDataloader import ReconstructionDataloader
from os import listdir
from os.path import join


class ReconstructionData(DataBase):

    def initialise_dataloader(self, is_train):
        if is_train:
            print('Initialise training dataloader')
            self.trn_dl = ReconstructionDataloader(self.trn_ds,
                                                   **self.trn_dl_params)
        else:
            print('Initialise validation dataloader')
            self.val_dl = ReconstructionDataloader(self.val_ds,
                                                   **self.val_dl_params)
