from ovseg.networks.recon_networks import get_operator
from ovseg.preprocessing.Reconstruction2dSimPreprocessing import Reconstruction2dSimPreprocessing
import os

op = get_operator(256)
prep = Reconstruction2dSimPreprocessing(op, num_photons= 2*10**6)

if not os.path.exists(os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed',
                                   'OV04', 'pod_full', 'projections')):
    prep.preprocess_raw_folders('OV04', preprocessed_name='pod_full')
