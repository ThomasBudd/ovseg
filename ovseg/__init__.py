from __future__ import absolute_import
from . import *
import os

try:
    ov_data_base = os.environ['OV_DATA_BASE']
except KeyError:
    raise KeyError('Envoironment variabll \'OV_DATA_BASE\' not found. Make '
                   'sure your system know where is it, because here all '
                   'raw and preprocessed data as well as the trained models '
                   'are kept.')
