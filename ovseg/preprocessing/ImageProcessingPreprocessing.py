import torch
import numpy as np
from ovseg.utils.torch_np_utils import check_type, stack
from ovseg.utils.path_utils import maybe_create_path
import os
from tqdm import tqdm
import nibabel as nib


class ImageProcessingPreprocessing(object):
    '''
    Just normalises the images gv and kicks out none [512, 512] images
    '''

    def __init__(self, window=[-1024, 1024]):
        self.window = window

    def preprocess_image(self, img):
        '''
        Simulation of 2d sinograms and windowing/rescaling of images
        If im is the image after rescaling (and windowing) and R the Ray transform we simulate as

            proj = -1/mu x log( Poisson(n_photons x exp(-mu R(im)))/n_photons )
        mu is just a scaling constant to
        '''
        # input img must be in HU
        if isinstance(img, np.ndarray):
            return (img.clip(*self.window) - self.window[0])/(self.window[1] - self.window[0])

        elif torch.is_tensor(img):
            return (img.clamp(*self.window) - self.window[0])/(self.window[1] - self.window[0])
        else:
            raise TypeError('Input of \'preprocess_image\' must be np array '
                            'or torch tensor. Got {}'.format(type(img)))

    def preprocess_volume(self, volume):
        # input volume must be in HU
        check_type(volume)
        if not len(volume.shape) == 3:
            raise ValueError('Preprocessing/simulation of projection data is '
                             'only implemented for 3d volumes. '
                             'Got shape {}'.format(len(volume.shape)))
        if not volume.shape[0] == 512 or not volume.shape[1] == 512:
            raise ValueError('Volume must be of shape (512, 512, nz). '
                             'Got {}'.format(volume.shape))
        return self.preprocess_image(volume)

    def preprocess_raw_folders(self, folders, preprocessed_name,
                               data_name=None,
                               im_folder_name='images_win_norm',
                               save_fp16=False):

        dtype = np.float16 if save_fp16 else np.float32

        if isinstance(folders, str):
            folders = [folders]
        elif not isinstance(folders, (list, tuple)):
            raise TypeError('Input folders must be string, list or tuple of '
                            'strings. ')

        # get the base folders
        ov_data_base = os.environ['OV_DATA_BASE']
        raw_data_base = os.path.join(ov_data_base, 'raw_data')
        raw_folders = os.listdir(raw_data_base)

        # check the content of folders
        for folder in folders:
            if not isinstance(folder, str):
                raise TypeError('Input folders must be string, list or tuple of '
                                'strings. ')
            elif folder not in raw_folders:
                raise FileNotFoundError('Folder {} was not found in {}'
                                        ''.format(folder, raw_data_base))

        if data_name is None:
            data_name = '_'.join(sorted(folders))
        preprocessed_data_base = os.path.join(ov_data_base,
                                              'preprocessed',
                                              data_name,
                                              preprocessed_name)
        scans = []
        for folder in folders:
            imp = os.path.join(raw_data_base, folder, 'images')
            scans.extend([os.path.join(imp, scan) for scan in os.listdir(imp)])

        for f in [im_folder_name]:
            maybe_create_path(os.path.join(preprocessed_data_base, f))

        for scan in tqdm(scans):
            name = os.path.basename(scan)[:8]
            volume = nib.load(scan).get_fdata()
            try:
                im = self.preprocess_volume(volume)
                np.save(os.path.join(preprocessed_data_base,
                                     im_folder_name,
                                     name+'.npy'),
                        im.astype(dtype), allow_pickle=True)
            except ValueError:
                print('Skip {}. Got shape {}.'.format(name, volume.shape))
