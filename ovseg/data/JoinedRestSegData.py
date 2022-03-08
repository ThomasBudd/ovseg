from ovseg.data.DataBase import DataBase
from ovseg.data.JoinedRestSegDataloader import JoinedRestSegDataloader


class JoinedRestSegData(DataBase):

    def initialise_dataloader(self, is_train):
        if is_train:
            print('Initialise training dataloader')
            self.trn_dl = JoinedRestSegDataloader(self.trn_ds, **self.trn_dl_params)
        else:
            print('Initialise validation dataloader')
            self.val_dl = JoinedRestSegDataloader(self.val_ds, **self.val_dl_params)
