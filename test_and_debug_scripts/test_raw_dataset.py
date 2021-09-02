from ovseg.data.Dataset import raw_Dataset
import os
from ovseg.utils.io import read_dcms

rp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data')

# %% try some reading of dcms
data_tpl = read_dcms(os.path.join(rp, 'BARTS_dcm', 'ID_001_1'))
data_tpl = read_dcms(os.path.join(rp, 'OV04_dcm', '034', 'CT_20091014'))


# %%
BARTS_ds = raw_Dataset(os.path.join(rp, 'BARTS_dcm'))

data_tpl = BARTS_ds[0]


# %%
OV04_ds = raw_Dataset(os.path.join(rp, 'OV04_dcm'))

# %%
ApolloTCGA_ds = raw_Dataset(os.path.join(rp, 'ApolloTCGA'))
