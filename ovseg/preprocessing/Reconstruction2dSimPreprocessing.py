import torch
import numpy as np
from ovseg.utils.torch_np_utils import check_type, stack
from ovseg.utils.path_utils import maybe_create_path
from ovseg.data.Dataset import raw_Dataset
from os.path import join, isdir, exists
from os import environ, listdir
import os
try:
    from tqdm import tqdm
except ImportError:
    print('tqdm not installed. Not pretty progressing bars')
    tqdm = lambda x: x
import nibabel as nib


class Reconstruction2dSimPreprocessing(object):
    '''
    Does what the name comes from: Simulation of the 2d sinograms
    '''

    def __init__(self, operator, num_photons=None, mu_water=0.0192, window=None,
                 scaling=None):
        self.operator = operator
        self.num_photons = num_photons
        self.mu_water = mu_water
        self.window = window
        self.scaling = scaling
        if self.scaling is None:
            if self.window is None:
                self.scaling = [1000 / self.mu_water, - 1000]
            else:
                self.scaling = [self.window[1] - self.window[0], self.window[0]]

    def preprocess_image(self, img):
        '''
        Simulation of 2d sinograms and windowing/rescaling of images
        If im is the image after rescaling (and windowing) and R the Ray transform we simulate as

            proj = -1/mu x log( Poisson(n_photons x exp(-mu R(im)))/n_photons )
        mu is just a scaling constant to
        '''
        # input img must be in HU
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
            is_np = True
        elif not torch.is_tensor(img):
            raise TypeError('Input of \'preprocess_image\' must be np array '
                            'or torch tensor. Got {}'.format(type(img)))
        else:
            is_np = False

        if not len(img.shape) == 2:
            raise ValueError('Preprocessing/simulation of projection data is '
                             'only implemented for 2d images. '
                             'Got shape {}'.format(len(img.shape)))

        # window the image if we're doing this cheat
        if self.window is not None:
            img = img.clip(*self.window)

        # now scale everything
        img = (img - self.scaling[0]) / self.scaling[1]

        img = img.type(torch.float).to('cuda')

        proj = self.operator.forward(img)
        if self.num_photons is not None:
            proj_exp = torch.exp(-1 * proj)
            proj_exp = torch.poisson(proj_exp * self.num_photons) / self.num_photons
            proj = -1 * torch.log(proj_exp + 1e-6)

        if is_np:
            return proj.cpu().numpy(), img.cpu().numpy()
        else:
            return proj, img

    def preprocess_volume(self, volume):
        # input volume must be in HU
        check_type(volume)
        if not len(volume.shape) == 3:
            raise ValueError('Preprocessing/simulation of projection data is '
                             'only implemented for 3d volumes. '
                             'Got shape {}'.format(len(volume.shape)))
        if not volume.shape[1] == 512 or not volume.shape[2] == 512:
            raise ValueError('Volume must be of shape (512, 512, nz). '
                             'Got {}'.format(volume.shape))
        projs = []
        im_atts = []
        nz = volume.shape[-1]
        for z in range(nz):
            proj, im_att = self.preprocess_image(volume[z])
            projs.append(proj)
            im_atts.append(im_att)

        proj = stack(projs)
        im_att = stack(im_atts)

        return proj, im_att

    def preprocess_raw_folders(self, folders, preprocessed_name,
                               data_name=None,
                               proj_folder_name='projections',
                               im_folder_name='images',
                               save_as_fp16=True):
        if isinstance(folders, str):
            folders = [folders]
        elif not isinstance(folders, (list, tuple)):
            raise TypeError('Input folders must be string, list or tuple of '
                            'strings. ')

        dtype = np.float16 if save_as_fp16 else np.float32
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
        for f in [proj_folder_name, im_folder_name]:
            maybe_create_path(os.path.join(preprocessed_data_base, f))

        print('Creating datasets...')
        datasets = []
        for data_name in folders:
            print('Reading ' + data_name)
            raw_ds = raw_Dataset(join(environ['OV_DATA_BASE'], 'raw_data', data_name))
            datasets.append(raw_ds)

        for ds in datasets:
            for i in range(len(ds)):
                data_tpl = ds[i]
                name = data_tpl['scan']
                volume = data_tpl['image']
                try:
                    proj, im = self.preprocess_volume(volume)
                    np.save(os.path.join(preprocessed_data_base,
                                         proj_folder_name,
                                         name+'.npy'),
                            proj.astype(dtype), allow_pickle=True)
                    np.save(os.path.join(preprocessed_data_base,
                                         im_folder_name,
                                         name+'.npy'),
                            im.astype(dtype), allow_pickle=True)
                except ValueError:
                    print('Skip {}. Got shape {}.'.format(name, volume.shape))
