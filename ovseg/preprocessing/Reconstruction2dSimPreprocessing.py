import torch
import numpy as np
from ovseg.utils.torch_np_utils import check_type, stack
from ovseg.utils.path_utils import maybe_create_path
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

    def __init__(self, operator, num_photons=2*10**6, mu_water=0.0192, window=None):
        self.operator = operator
        self.num_photons = num_photons
        self.mu_water = mu_water
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

        # bring the image to the desired coordinates
        # the mu_scale makes sure that the number of photons adds the same noise regardless
        # of the image scaling
        if self.window is None:
            im_att = img/1000 * self.mu_water + self.mu_water
            mu_scale = 1
        else:
            im_att = (img.clip(*self.window) - self.window[0])/(self.window[1] - self.window[0])
            mu_scale = (self.window[1] - self.window[0]) * self.mu_water / 1000

        im_att = im_att.type(torch.float).clamp(0).to('cuda')

        clean_proj = self.operator.forward(im_att)
        clean_proj_exp = torch.exp(-1 * mu_scale * clean_proj)
        noisy_proj = torch.poisson(clean_proj_exp * self.num_photons) / \
            self.num_photons
        noisy_proj = -1 * torch.log(noisy_proj + 1e-6) / mu_scale

        if is_np:
            return noisy_proj.cpu().numpy(), im_att.cpu().numpy()
        else:
            return noisy_proj, im_att

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
        projs = []
        im_atts = []
        nz = volume.shape[-1]
        for z in range(nz):
            proj, im_att = self.preprocess_image(volume[..., z])
            projs.append(proj)
            im_atts.append(im_att)

        proj = stack(projs, -1)
        im_att = stack(im_atts, -1)

        return proj, im_att

    def preprocess_raw_folders(self, folders, preprocessed_name,
                               data_name=None,
                               proj_folder_name='projections',
                               im_folder_name='images',
                               save_as_fp16=False):
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
            elif not folder in raw_folders:
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

        for f in [proj_folder_name, im_folder_name]:
            maybe_create_path(os.path.join(preprocessed_data_base, f))

        for scan in tqdm(scans):
            name = os.path.basename(scan)[:8]
            volume = nib.load(scan).get_fdata()
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
