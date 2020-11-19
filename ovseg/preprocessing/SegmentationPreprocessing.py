import numpy as np
import torch
from ovseg.utils.interp_utils import change_img_pixel_spacing,\
    change_sample_pixel_spacing
from ovseg.utils.torch_np_utils import check_type, stack
from ovseg.utils.io import read_nii_files, load_pkl, save_pkl
from ovseg.utils.path_utils import my_listdir, maybe_create_path
from os.path import join, isdir, exists
from os import remove, environ
from tqdm import tqdm


class SegmentationPreprocessing(object):
    '''
    Class that is responsible for performing preprocessing of segmentation data
    This class expects
        1) single channel images
        2) non overlappting segmentation in integer encoding
    If the corresponding flags are set we perform
         1) resizing to change the pixel spacing to target_spacing
         2) additional downsampling by factor 2, 3, or 4
         3) windowing/clipping of image values
         4) scaling of the gray values x --> (x-scaling[1])/scaling[0]
    Images will be resampled with first or third order by default.
    Segementations are decoded to one hot vectors resampled by trilinear
    interpolation and decoded to integer encoding by argmax
    '''

    def __init__(self,
                 data_name,
                 preprocessed_name,
                 apply_resizing=True,
                 target_spacing=None,
                 apply_windowing=True,
                 window=None,
                 apply_scaling=True,
                 scaling=None,
                 apply_downsampling=False,
                 downsampling_fac=None,
                 use_only_fg_scans=True,
                 use_only_classes=None,
                 try_preprocess_volume_in_torch=True,
                 seg_channels=[1],
                 use_torch_for_data_conversion=True,
                 percentiles_fg_voxel=[0.5, 99.5],
                 **kwargs):

        self.data_name = data_name
        self.preprocessed_name = preprocessed_name
        self.apply_resizing = apply_resizing
        self.target_spacing = target_spacing
        self.apply_windowing = apply_windowing
        self.window = window
        self.apply_scaling = apply_scaling
        self.scaling = scaling
        self.apply_downsampling = apply_downsampling
        self.downsampling_fac = downsampling_fac
        self.use_only_fg_scans = use_only_fg_scans
        self.use_only_classes = use_only_classes
        self.try_preprocess_volume_in_torch = try_preprocess_volume_in_torch
        self.use_torch_for_data_conversion = use_torch_for_data_conversion
        self.seg_channels = seg_channels
        self.percentiles_fg_voxel = percentiles_fg_voxel

        self.path_to_preprocessing_file = join(environ['OV_DATA_BASE'],
                                               'preprocessed',
                                               self.data_name,
                                               self.preprocessed_name,
                                               'preprocessing_parameters.pkl')

        self.attributes_loaded = False

        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.apply_resizing and target_spacing is None:
            if exists(self.path_to_preprocessing_file):
                print('Not all preprocessing parameters were initialised. '
                      'Loading parameters from folder.')
                self.load_attributes(False)
            else:
                print('Expected \'target_spacing\' to be list or tuple.'
                      'Got None instead.\n')
        else:
            self.target_spacing = np.array(self.target_spacing)
        if self.apply_windowing and window is None:
            if exists(self.path_to_preprocessing_file):
                print('Not all preprocessing parameters were initialised. '
                      'Loading parameters from folder.')
                self.load_attributes(False)
            else:
                print('Expected \'window\' to be list or tuple of length 2.'
                      ' Got None instead. Use plan_preprocessing to infere '
                      'this parameter or load them.')
        else:
            self.window = np.array(self.window)
        if self.apply_scaling and scaling is None:
            if exists(self.path_to_preprocessing_file):
                print('Not all preprocessing parameters were initialised. '
                      'Loading parameters from folder.')
                self.load_attributes(False)
            else:
                print('Expected \'scaling\' to be list or tuple of '
                      'length 2. Got None instead. Use plan_preprocessing to '
                      'infere this parameter or load them.')
        else:
            self.scaling = np.array(self.scaling)

        if self.apply_downsampling and self.apply_resizing:
            print('Downsampling and resizing used as preprocessing. Expecting '
                  'the downsampling factor to be included in the '
                  'target_spacing.')

        # check if the right downsampling factor was given.
        if self.apply_downsampling:
            if self.downsampling_fac is None:
                raise TypeError('downsampling_fac was not initialised.')
            elif not isinstance(self.downsampling_fac, (list, tuple,
                                                        np.ndarray)):
                raise TypeError('downsampling_fac must be list, tuple or '
                                'np array')
            elif not len(self.downsampling_fac) == 3:
                raise ValueError('downsampling_fac must be of len 3, one '
                                 'factor for each axis.')

        # when inheriting from this please append any other important
        # parameters that define the preprocessing
        self.preprocessing_parameters = ['apply_resizing', 'target_spacing',
                                         'apply_windowing', 'window',
                                         'apply_scaling', 'scaling',
                                         'apply_downsampling',
                                         'downsampling_fac',
                                         'use_only_classes',
                                         'use_only_fg_scans',
                                         'try_preprocess_volume_in_torch',
                                         'seg_channels',
                                         'percentiles_fg_voxel']

        # points to the raw data, we need it for infering parameters
        # or contering data
        self.raw_data_folder = join(environ['OV_DATA_BASE'], 'raw_data',
                                    self.data_name)

        self.preprocessed_folder = join(environ['OV_DATA_BASE'],
                                        'preprocessed',
                                        self.data_name,
                                        self.preprocessed_name)

    def get_attributes(self):
        data = {key: self.__getattribute__(key) for key in
                self.preprocessing_parameters}
        return data

    def save_attributes(self):
        if exists(self.path_to_preprocessing_file):
            print('Warning: Found existing pickle file of preprocessing '
                  'attributes. Removing this one.\n')
            remove(self.path_to_preprocessing_file)
        data = self.get_attributes()
        save_pkl(data, self.path_to_preprocessing_file)

    def load_attributes(self, overwrite_set_parameters=True):
        if not exists(self.path_to_preprocessing_file):
            raise FileNotFoundError('Couldn\'t load preprocessing parameters '
                                    'at path ' +
                                    self.path_to_preprocessing_file +
                                    '. No such file.')

        # if not we can load the file
        data = load_pkl(self.path_to_preprocessing_file)
        for key in data:
            # check if we want to overwrite the not None parameters
            if not overwrite_set_parameters and \
                    self.__getattribute__(key) is not None:
                continue

            # else we set the attribute
            self.__setattr__(key, data[key])
            self.attributes_loaded = True

    def preprocess_image(self, img, is_seg, spacing=None):
        '''
        Preprocessed a single image (channel)

        Parameters
        ----------
        img : np.ndarray or torch.tensor,
            shape (nx, ny, nz) or (1, nx, ny, nz)
        is_seg : bool
            if the input is a segmentation map
        spacing : list, tuple, np.ndarray
            voxel spacing in mm
        Returns:
            preprocessed image
        -------
        None.

        '''
        is_np, is_torch = check_type(img)
        shape_in = np.array(img.shape)
        if len(shape_in) not in [2, 3]:
            raise ValueError('Input must be of shape (nx, ny) or '
                             '(nx, ny, nz). Got {}'.fomat(shape_in))
        # first we case the image to float 32
        if is_np:
            img = img.astype(np.float32)
        else:
            img = img.type(torch.float32)

        # next resizing
        # first let's check for spacing
        if self.apply_resizing or self.apply_downsampling():
            if spacing is None:
                raise TypeError('Input spacing must be given, when '
                                'apply_resizing=True or '
                                'apply_downsampling=True.')
            elif not isinstance(spacing, (list, tuple, np.ndarray)):
                raise TypeError('Spacing must be of type '
                                'list, tuple or np.ndarray')
        # the preprocessing of segmentations and images are quite different
        # we only apply resizing and downsampling in here
        if is_seg:

            # if not there's nothing to be done
            if not self.apply_resizing and not self.apply_downsampling:
                return img

            # convert to one hot encoding
            nclasses = int(img.max())
            img = stack([img == c for c in range(nclasses+1)])
            if is_np:
                img = img.astype(np.float32)
            else:
                img = img.type(torch.float)

            # for interpolation
            idim = len(shape_in)
            order = 1

            # first resizing
            if self.apply_resizing:
                # the img extended to a sample, with one hot vectors in each
                # channel
                img = change_sample_pixel_spacing(img, spacing[:idim],
                                                  self.target_spacing[:idim],
                                                  orders=(nclasses+1)*[order])

            # if not downsampling
            if self.apply_downsampling and not self.apply_resizing:
                # the exact number don't matter here. The resizing factor
                # is computed relative anyways.
                img = change_sample_pixel_spacing(img, np.ones(idim),
                                                  self.downsampling_fac[:idim],
                                                  orders=(nclasses+1)*[order])

            # bring back to integer encoding
            if is_np:
                return np.argmax(img, 0).astype(np.float32)
            else:
                return torch.argmax(img, 0).type(torch.float)

        # segmentations are done now! Let's consider only images here
        idim = len(shape_in)
        order = 3 if is_np else 1
        if self.apply_resizing:
            img = change_img_pixel_spacing(img, spacing[:idim],
                                           self.target_spacing[:idim],
                                           order=order)

        # if not downsampling
        if self.apply_downsampling and not self.apply_resizing:
            # the exact number don't matter here. The resizing factor
            # is computed relative anyways.
            img = change_img_pixel_spacing(img, np.ones(idim),
                                           self.downsampling_fac[:idim],
                                           order=order)

        # now windowing
        if self.apply_windowing:
            if is_np:
                img = img.clip(self.window[0], self.window[1])
            else:
                img = img.clamp(self.window[0], self.window[1])

        # last but not least the rescaling
        if self.apply_scaling:
            img = (img - self.scaling[1])/self.scaling[0]

        return img

    def preprocess_sample(self, sample, spacing=None):
        '''
        Preprocesses a sample. Assumes first channel to be the image and
        the following to be segmentation maps

        Parameters
        ----------
        sample : np.ndarray or torch.tensor
            shape: [channels, nx, ny(, nz)]
        spacing : dict
            contains information on spacing

        Returns
        -------
        None.

        '''
        check_type(sample)
        nch = sample.shape[0]
        is_seg = [i in self.seg_channels for i in range(nch)]
        return stack([self.preprocess_image(sample[c], is_seg[c], spacing)
                      for c in range(nch)])

    def preprocess_batch(self, batch, spacings=None):
        '''
        Preprocesses a sample. Assumes first channel to be the image and
        the following to be segmentation maps

        Parameters
        ----------
        sample : np.ndarray or torch.tensor
            shape: [channels, nx, ny(, nz)]
        spacing : dict
            contains information on spacing

        Returns
        -------
        None.

            '''
        if not (isinstance(batch, (tuple, list, np.ndarray))
                or torch.is_tensor(batch)):
            raise ValueError('Input must be batch items in a '
                             'list, tuple, np.ndarray or torch.tensor.')
        bs = len(batch)
        if spacings is None:
            spacings = [None for _ in range(bs)]
        assert len(batch) == len(spacings)
        return [self.preprocess_sample(b, s) for b, s in zip(batch, spacings)]

    def preprocess_volume(self, volume, spacing=None):
        '''
        Preprocesses a full volume

        Parameters
        ----------
        volume : 3d or 4d tensor
            np array or torch tensor, channels first
        spacing : list, tuple or np.ndarray
            voxel spacing in real world units (mm)

        Returns
        -------
        None.

        '''
        is_np, _ = check_type(volume)
        if self.try_preprocess_volume_in_torch and is_np:
            volume = torch.from_numpy(volume).to(self.dev)
        elif not self.try_preprocess_volume_in_torch and not is_np:
            # if try_preprocess_volume_in_torch is False we have probably
            # gone out of ram another time. Let's force to do it on the CPU
            volume = volume.cpu().numpy()
        shape = np.array(volume.shape)

        # we process the volume as a sample
        if len(shape) == 3:
            if is_np:
                volume = volume[np.newaxis]
            else:
                volume = volume.unsqueeze(0)

        # here the exciting part! We preprocess the sample
        # trying to do it on the GPU
        try:
            volume = self.preprocess_sample(volume, spacing)
        except RuntimeError:
            print('Problem while preprocessing a full volume. Probably CUDA '
                  'got out of RAM. Moving to the CPU (this will be slow).')
            self.try_preprocess_volume_in_torch = False
            # remember this for later!
            self.save_attributes()
            volume = volume.cpu().numpy()
            volume = self.preprocess_sample(volume, spacing)

        # now do the inverse of the preparation
        if len(shape) == 3:
            volume = volume[0]

        if is_np and torch.is_tensor(volume):
            # if the input is numpy the output will be as well even if we
            # do the preprocessing on the GPU
            volume = volume.cpu().numpy()
        elif not is_np and isinstance(volume, np.ndarray):
            # same with torch vice versa
            volume = torch.tensor(volume).to(self.dev)

        return volume

    def __call__(self, volume, spacing=None):
        return self.preprocess_volume(volume, spacing)

    def _find_im_and_lb_path(self, raw_data_folder=None):

        if raw_data_folder is None:
            # if no folder is given we choose by default our
            # path
            raw_data_folder = self.raw_data_folder

        subdirs = [d for d in my_listdir(self.raw_data_folder) if
                   isdir(join(self.raw_data_folder, d))]
        if 'labels' in subdirs:
            lbp = join(self.raw_data_folder, 'labels')
        elif 'labelsTr' in subdirs:
            lbp = join(self.raw_data_folder, 'labelsTr')
        else:
            raise FileNotFoundError('Didn\'t find label folder in ' +
                                    self.raw_data_folder
                                    + '. Label folders are supposed to be '
                                    'named \'labels\' or \'labelsTr\'.')
        if 'images' in subdirs:
            imp = join(self.raw_data_folder, 'images')
        elif 'imagesTr' in subdirs:
            imp = join(self.raw_data_folder, 'imagesTr')
        else:
            raise FileNotFoundError('Didn\'t find image folder in ' +
                                    self.raw_data_folder
                                    + '. Image folders are supposed to be '
                                    'named \'images\' or \'imagesTr\'.')
        return imp, lbp

    def preprocess_raw_data_folder(self, raw_data_folder=None):

        # first let's get the image and label path
        imp, lbp = self._find_im_and_lb_path(raw_data_folder)
        # now let's create the output folders
        for f in ['images', 'labels', 'spacings', 'orig_shapes']:
            maybe_create_path(join(self.preprocessed_folder, f))
        # now let's get all the cases
        cases = [case[:-7] for case in my_listdir(lbp)
                 if case.endswith('.nii.gz')]

        # Let's quickly store the parameters so we can check later
        # what we've done here.
        self.save_attributes()

        # here is the fun
        for case in tqdm(cases):

            # first let's take the image nii files that start with the same
            # case_id as the label
            nii_files = [join(imp, nii_file) for nii_file in my_listdir(imp)
                         if nii_file.endswith('.nii.gz')
                         and nii_file.startswith(case)]

            # ATM we're only doing one image per label. For MR we might have
            # mulitple
            if not len(nii_files) == 1:
                raise FileNotFoundError('Assumend to find exactly one images '
                                        'starting with ')

            # the list of nii files is needed to read them all in a sample
            nii_files.append(join(lbp, case+'.nii.gz'))
            sample, spacing = read_nii_files(nii_files)

            # reduce the lesion to the class we're intersted in in this case
            sample[1] = self._remove_classes(sample[1])

            # turns out only using scans that do have show at least one
            # foreground class perform better best
            if self.use_only_fg_scans and sample[1].max() == 0:
                continue
            else:

                # if we do have fg clases we save the boy
                orig_shape = sample[0].shape
                if self.use_torch_for_data_conversion:
                    # using torch is very fast, but we're limited to trilinear
                    # interpolation
                    sample = torch.tensor(sample).to('cuda')

                # the most important part
                sample = self.preprocess_sample(sample, spacing)
                if self.use_torch_for_data_conversion:
                    sample = sample.cpu().numpy()

                # save them in the right dtype. int8 is way faster to load
                im, lb = sample[0].astype(np.float32),\
                    sample[1].astype(np.int8)

                # save all the information in seperate folders
                for arr, folder in [[im, 'images'], [lb, 'labels'],
                                    [orig_shape, 'orig_shapes'],
                                    [spacing, 'spacings']]:
                    np.save(join(self.preprocessed_folder, folder, case), arr)
        print('Preprocessing done!')

    def _remove_classes(self, lb):

        # remove classes from the label for the case we don't want to segment
        # all classes with one network
        if self.use_only_classes is None:
            return lb
        else:
            lb_new = np.zeros_like(lb)
            for i, c in enumerate(self.use_only_classes):
                lb_new[lb == c] = i+1
            return lb_new

    def plan_preprocessing_from_raw_folder(self):

        print('Infering preprocessing parameters from '+self.raw_data_folder)
        # first let's get the image and label path
        imp, lbp = self._find_im_and_lb_path()
        # now let's get all the cases
        cases = [case[:-7] for case in my_listdir(lbp)
                 if case.endswith('.nii.gz')]
        # foreground vals for windowing
        fg_cvals = []
        # mean and std for scaling
        mean = 0
        mean2 = 0
        fg_cases = 0
        # spacings for resampling
        spacings = []
        # here is the fun
        for case in tqdm(cases):
            # first let's take the image nii files
            nii_files = [join(imp, nii_file) for nii_file in my_listdir(imp)
                         if nii_file.endswith('.nii.gz')
                         and nii_file.startswith(case)]
            if not len(nii_files) == 1:
                raise FileNotFoundError('Assumend to find exactly one images '
                                        'starting with ')
            nii_files.append(join(lbp, case+'.nii.gz'))
            # read files
            sample, spacing = read_nii_files(nii_files)
            sample[1] = self._remove_classes(sample[1])
            im, lb = sample
            # store spacing
            spacings.append(spacing)
            if lb.max() > 0:
                fg_cval = im[lb > 0].astype(float)
                fg_cvals.extend(fg_cval.tolist())
                mean += np.mean(fg_cval)
                mean2 += np.mean(fg_cval**2)
                fg_cases += 1
        mean = mean/fg_cases
        mean2 = mean2/fg_cases
        std = np.sqrt(mean2 - mean**2)
        if self.apply_scaling and self.scaling is None:
            self.scaling = np.array([std, mean])
            print('Scaling: ({:.4f}, {:.4f})'.format(*self.scaling))
        if self.apply_resizing and self.target_spacing is None:
            self.target_spacing = np.median(np.stack(spacings), 0)
            if self.apply_downsampling:
                # when downsampling we just multiply the factor with the target
                # spacing
                if self.downsampling_fac is not None:
                    self.target_spacing *= np.array(self.downsampling_fac)
            print('Spacing: ({:.4f}, {:.4f}, {:.4f})'.
                  format(*self.target_spacing))
        if self.apply_windowing and self.window is None:
            self.window = np.percentile(fg_cvals, self.percentiles_fg_voxel)
            print('Window: ({:.4f}, {:.4f})'.format(*self.window))
