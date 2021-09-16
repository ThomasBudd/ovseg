from __future__ import absolute_import
from . import *
import os

try:
    ov_data_base = os.environ['OV_DATA_BASE']
except KeyError:
    raise KeyError('Envoironment variabll \'OV_DATA_BASE\' not found. Make '
                   'sure your system know where is it, because here all '
                   'raw data, predictions as well as the trained models '
                   'are kept.')

if 'OV_PREPROCESSED' in os.environ:
    OV_PREPROCESSED = os.environ['OV_PREPROCESSED']
else:
    OV_PREPROCESSED = os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed')
