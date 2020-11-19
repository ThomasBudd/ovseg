from ovseg.data.DataBase import DataBase
from ovseg.data.JoinedDataloader import JoinedDataloader


class JoinedData(DataBase):

    def initialise_dataloader(self, is_train):
        if is_train:
            print('Initialise training dataloader')
            self.trn_dl = JoinedDataloader(self.trn_ds, **self.trn_dl_params)
        else:
            print('Initialise validation dataloader')
            self.val_dl = JoinedDataloader(self.val_ds, **self.val_dl_params)
