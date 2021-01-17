from ovseg.preprocessing.Reconstruction2dSimPreprocessing import Reconstruction2dSimPreprocessing
from ovseg.networks.recon_networks import get_operator

operator = get_operator()
preprocessing = Reconstruction2dSimPreprocessing(operator)
preprocessing.preprocess_raw_folders(['OV04', 'BARTS', 'ApolloTCGA'],
                                     'pod_default',
                                     data_name='OV04',
                                     proj_folder_name='projections',
                                     im_folder_name='images_att')
