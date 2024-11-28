from __future__ import absolute_import
from . import *
import os

if 'OV_DATA_BASE' not in os.environ:
    OV_DATA_BASE = os.path.join(os.path.dirname(__file__), 'ov_data_base')
    print('No environment variable OV_DATA_BASE specified. Model weights,  '
          'the resulting segmentations etc. will be stored at '
          f'{OV_DATA_BASE}.')
    os.environ['OV_DATA_BASE'] = OV_DATA_BASE
else:    
    OV_DATA_BASE = os.environ['OV_DATA_BASE']

os.makedirs(OV_DATA_BASE, exist_ok=True)

if 'OV_PREPROCESSED' in os.environ:
    OV_PREPROCESSED = os.environ['OV_PREPROCESSED']
else:
    OV_PREPROCESSED = os.path.join(OV_DATA_BASE, 'preprocessed')
