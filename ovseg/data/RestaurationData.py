from ovseg.data.DataBase import DataBase
from ovseg.data.RestaurationDataloader import RestaurationDataloader

class RestaurationData(DataBase):

    def initialise_dataloader(self, is_train):
        if is_train:
            print('Initialise training dataloader')
            self.trn_dl = RestaurationDataloader(self.trn_ds, **self.trn_dl_params)
        else:
            print('Initialise validation dataloader')
            self.val_dl = RestaurationDataloader(self.val_ds, **self.val_dl_params)
