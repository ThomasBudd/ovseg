import torch
import numpy as np
from ovseg.utils.torch_np_utils import check_type, stack
from ovseg.utils.path_utils import maybe_create_path
from ovseg.data.Dataset import raw_Dataset
from os.path import join, isdir, exists, basename
from os import environ, listdir
from torch_radon import RadonFanbeam
import os
try:
    from tqdm import tqdm
except ImportError:
    print('tqdm not installed. Not pretty progressing bars')
    tqdm = lambda x: x
from time import sleep
from skimage.transform import resize
from ovseg.utils.dict_equal import dict_equal, print_dict_diff
from ovseg.utils.io import load_pkl, save_pkl, save_txt


class Restauration2dSimPreprocessing(object):
    '''
    Does what the name comes from: Simulation of the 2d sinograms
    '''

    def __init__(self, n_angles=500, source_distance=600, det_count=736, det_spacing=1.0,
                 mu_water=0.0192, window=None, scaling=None,
                 fbp_filter='ramp', apply_z_resizing=True, target_z_spacing=None,
                 bowtie_filt=None, dose_level=1.0):
        self.n_angles = n_angles
        self.source_distance = source_distance
        self.det_count = det_count
        self.det_spacing = det_spacing
        self.mu_water = mu_water
        self.window = window
        self.scaling = scaling
        self.fbp_filter = fbp_filter
        self.apply_z_resizing = apply_z_resizing
        self.target_z_spacing = target_z_spacing
        self.dose_level = dose_level
        
        if self.apply_z_resizing and self.target_z_spacing is None:
            raise ValueError('target_z_spacing not set.')

        self.operator = RadonFanbeam(512,
                                     np.linspace(0,2*np.pi, self.n_angles),
                                     source_distance=self.source_distance, 
                                     det_count=self.det_count,
                                     det_spacing=self.det_spacing)
        self.preprocessing_parameters = ['n_angles', 'source_distance', 'det_count', 'det_spacing',
                                         'window', 'scaling',
                                         'fbp_filter', 'apply_z_resizing', 'target_z_spacing',
                                         'bowtie_filt', 'dose_level']

        if bowtie_filt is None:
            bowtie_filt = np.load('default_photon_stats.npy')/64
        # ADD OTHER FILTERS HERE?

        '''
        bowtie filter
        '''
        if isinstance(bowtie_filt, np.ndarray):
            bowtie_filt = torch.from_numpy(bowtie_filt)
            bowtie_filt = bowtie_filt.squeeze()
        elif not torch.is_tensor(bowtie_filt):
            raise TypeError('The bowtie filter must be a numpy array or torch tensor. Got {}'
                            ''.format(type(bowtie_filt)))

        
        if not len(bowtie_filt.shape) == 1:
            raise ValueError('The bowtie filter must be a 1D array'
                             'Got shape {}'.format(len(bowtie_filt.shape)))
            
        if not bowtie_filt.size()[0] == self.det_count:
            raise ValueError('The bowtie filter must have the same size as the number of detector '
                             'pixels {}. Got size {}'.format(self.det_count, bowtie_filt.size()[0]))

        self.bowtie_filt = bowtie_filt

    def maybe_save_preprocessing_parameters(self, outfolder, ext=None):
        param_name = 'restauration_parameters'
        if ext is not None:
            param_name = param_name + '_' + ext
        outfile = join(outfolder, 'restauration_parameters.pkl')
        data = {key: self.__getattribute__(key) for key in
                self.preprocessing_parameters}
        if exists(outfile):
            data_pkl = load_pkl(outfile)
            if dict_equal(data_pkl, data):
                return
            else:
                print_dict_diff(data_pkl, data, 'pkl paramters', 'given paramters')
                raise RuntimeError('Found not matching prerpocessing parameters in '+outfolder+'.')
        else:
            save_pkl(data, outfile)
            save_txt(data, outfile[:-4])

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

        # we're ingoring HU < 1000
        img = img.type(torch.float).clip(-1000)
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'

        # rescale from HU to linear attenuation
        img_linatt = (img + 1000) / 1000 * self.mu_water
        img_linatt = img_linatt.type(torch.float).to(dev)

        # rescale from HU to linear attenuation
        bowtie_filt = self.dose_level * self.bowtie_filt.to(dev)
        proj = self.operator.forward(img_linatt)
        proj = torch.exp(-proj) 
        proj = torch.poisson(bowtie_filt.expand_as(proj)*proj)*(1/bowtie_filt.expand_as(proj))
        sinogram_noisy = -torch.log(1e-6 + proj)
        fbp_linatt = self.operator.backprojection(self.operator.filter_sinogram(sinogram_noisy))
        fbp = 1000 * (fbp_linatt - self.mu_water) / self.mu_water
        
        # now windowing and recaling
        if self.window is not None:
            img = img.clip(*self.window)
            fbp = fbp.clip(*self.window)
        if self.scaling is not None:
            img = (img - self.scaling[1]) / self.scaling[0]
            fbp = (fbp - self.scaling[1]) / self.scaling[0]

        if is_np:
            return fbp.cpu().numpy(), img.cpu().numpy()
        else:
            return fbp, img

    def preprocess_volume(self, volume, spacing=None):
        # input volume must be in HU
        is_np, _ = check_type(volume)
        if not len(volume.shape) == 3:
            raise ValueError('Preprocessing/simulation of projection data is '
                             'only implemented for 3d volumes. '
                             'Got shape {}'.format(len(volume.shape)))
        if not volume.shape[1] == 512 or not volume.shape[2] == 512:
            raise ValueError('Volume must be of shape (nz, 512, 512). '
                             'Got {}'.format(volume.shape))

        if self.apply_z_resizing:
            # resample the z axis, the xy plane should be 512 x 512
            if spacing is None:
                raise ValueError('spacing must be given when resizing the z axis')
            nz_new = int(volume.shape[0] * spacing[0] / self.target_z_spacing + 0.5)
            if is_np:
                volume = resize(volume, [nz_new, 512, 512], order=1)
            else:
                if torch.cuda.is_available():
                    volume = volume.cuda()
                volume = torch.nn.functional.interpolate(volume, [nz_new, 512, 512],
                                                         mode='trilinear')
        fbps = []
        imgs = []
        nz = volume.shape[0]
        for z in range(nz):
            fbp, img = self.preprocess_image(volume[z])
            fbps.append(fbp)
            imgs.append(img)

        fbp = stack(fbps)
        img = stack(imgs)

        return fbp, img

    def preprocess_raw_folders(self, folders, preprocessed_name,
                               data_name=None,
                               fbp_folder_name='fbps',
                               im_folder_name='images_restauration',
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
        print('Storing new simulations in {}, {} at {}'.format(fbp_folder_name,
                                                               im_folder_name,
                                                               preprocessed_data_base))
        for f in [fbp_folder_name, im_folder_name]:
            maybe_create_path(os.path.join(preprocessed_data_base, f))

        if fbp_folder_name.startswith('fbp_'):
            ext = fbp_folder_name[4:]
        else:
            ext = None

        self.maybe_save_preprocessing_parameters(preprocessed_data_base, ext)

        print('Creating datasets...')
        datasets = []
        for data_name in folders:
            print('Reading ' + data_name)
            raw_ds = raw_Dataset(join(environ['OV_DATA_BASE'], 'raw_data', data_name))
            datasets.append(raw_ds)

        for ds in datasets:
            print(basename(ds.raw_path))
            sleep(0.5)
            for i in tqdm(range(len(ds))):
                data_tpl = ds[i]
                name = data_tpl['scan']
                volume = data_tpl['image']
                spacing = data_tpl['spacing']
                try:
                    fbp, im = self.preprocess_volume(volume, spacing)
                    np.save(os.path.join(preprocessed_data_base,
                                         fbp_folder_name,
                                         name+'.npy'),
                            fbp.astype(dtype), allow_pickle=True)
                    np.save(os.path.join(preprocessed_data_base,
                                         im_folder_name,
                                         name+'.npy'),
                            im.astype(dtype), allow_pickle=True)
                except ValueError:
                    print('Skip {}. Got shape {}.'.format(name, volume.shape))
