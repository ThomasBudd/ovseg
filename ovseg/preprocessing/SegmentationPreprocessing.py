import numpy as np
import torch
from ovseg.utils.interp_utils import change_img_pixel_spacing,\
    change_sample_pixel_spacing
from ovseg.utils.torch_np_utils import check_type, stack
from ovseg.utils.io import read_nii_files, load_pkl, save_pkl
from ovseg.utils.path_utils import my_listdir, maybe_create_path
from os.path import join, isdir, exists
from os import remove
import pickle
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
                 **kwargs):

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

        if self.apply_scaling and target_spacing is None:
            print('Expected \'target_spacing\' to be list or tuple.'
                  'Got None instead. Use plan_preprocessing to infere '
                  'this parameter or load them.')
        else:
            self.target_spacing = np.array(self.target_spacing)
        if self.apply_windowing and window is None:
            print('Expected \'window\' to be list or tuple of length 2. Got'
                  ' None instead. Use plan_preprocessing to infere '
                  'this parameter or load them.')
        else:
            self.window = np.array(self.window)
        if self.apply_scaling and scaling is None:
            print('Expected \'scaling\' to be list or tuple of '
                  'length 2. Got None instead. Use plan_preprocessing to '
                  'infere this parameter or load them.')
        else:
            self.scaling = np.array(self.scaling)

        if self.apply_downsampling and \
                self.downsampling_fac not in [2, 3, 4]:
            raise ValueError('Downsampling as preprocessing is only '
                             'implemented for the factors 2, 3 or 4.')

        # when inheriting from this please append any other important
        # parameters that define the preprocessing
        self.preprocessing_parameters = ['apply_resizing', 'target_spacing',
                                         'apply_windowing', 'window',
                                         'apply_scaling', 'scaling',
                                         'apply_downsampling',
                                         'downsampling_fac',
                                         'use_only_classes',
                                         'use_only_fg_scans']

    def _downsample_img_fac_2(self, img, downsample_z=False):
        ndims = len(img.shape)
        if ndims not in [2, 3]:
            raise ValueError('Expected 2d or 3d image, but got {}d'.
                             format(ndims))
        end = np.array(img.shape) // 2 * 2
        # x axis
        img = (img[:end[0]:2] + img[1:end[0]:2]) / 2
        # y axis
        img = (img[:, :end[1]:2] + img[:, 1:end[1]:2]) / 2
        if ndims == 3 and downsample_z:
            img = (img[:, :, :end[2]:2] + img[:, :, 1:end[2]:2]) / 2
        return img

    def _downsample_img_fac_3(self, img, downsample_z=False):
        ndims = len(img.shape)
        if ndims not in [2, 3]:
            raise ValueError('Expected 2d or 3d image, but got {}d'.
                             format(ndims))
        # x axis
        img = (img[::3] + img[1::3] + img[2::3]) / 2
        # y axis
        img = (img[:, ::3] + img[:, 1::3] + img[:, 2::3]) / 2
        if ndims == 3 and downsample_z:
            img = (img[:, :, ::3] + img[:, :, 1::3] + img[:, :, 2::3]) / 2
        return img

    def _downsample_img(self, img, spacing):

        r = np.max(spacing)/np.min(spacing)

        if self.downsampling_fac == 2:
            downsample_z = r < 4/3
            return self._downsample_img_fac_2(img, downsample_z)
        if self.downsampling_fac == 3:
            downsample_z = r < 3/2
            return self._downsample_img_fac_3(img, downsample_z)
        if self.downsampling_fac == 4:
            downsample_z1 = r < 4/3
            img = self._downsample_img_fac_2(img, downsample_z1)
            downsample_z2 = r < 8/3
            img = self._downsample_img_fac_2(img, downsample_z2)
        return img

    def save_attributes(self, outfolder):
        outfile = join(outfolder, 'preprocessing_parameters.pkl')
        if exists(outfile):
            print('Warning: Found existing pickle file of preprocessing '
                  'attributes. Removing this one.')
            remove(outfile)
        data = {key: self.__getattribute__(key) for key in
                self.preprocessing_parameters}
        save_pkl(data, outfile)

    def check_preprocessing(self, outfolder):
        pklfile = join(outfolder, 'preprocessing_parameters.pkl')
        # does the file even exsit?
        if not exists(pklfile):
            print('No pickel file with preprocessing parameters found at '
                  + outfolder)
            return False
        # load it!
        data = load_pkl(pklfile)
        # do the keys match?
        for key in self.preprocessing_parameters:
            if key not in data.keys():
                print('key ' + key + ' was not found in preprocessing '
                      'parameter file.')
                return False
        # not check if the keys are equal
        for key in self.preprocessing_parameters:
            item = self.__getattribute__(key)
            if data[key] != item:
                print('Found not matching items for key' + key)
                return False
        print('Preprocessing parameters match.')
        return True

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
            # first resizing
            if self.apply_resizing:
                idim = len(shape_in)
                order = 1
                # the img is a sample, resize all channels
                img = change_sample_pixel_spacing(img, spacing[:idim],
                                                  self.target_spacing[:idim],
                                                  orders=(nclasses+1)*[order])
            # now potentiall downsampling
            if self.apply_downsampling:
                sp = self.target_spacing if self.apply_resizing else\
                    spacing
                img = stack([self._downsample_img(img[c], sp) for c in
                             range(nclasses+1)])

            # bring back to integer encoding
            if is_np:
                return np.argmax(img, 0).astype(np.float32)
            else:
                return torch.argmax(img, 0).type(torch.float)

        # segmentations are done now! Let's consider only images here
        if self.apply_resizing:
            idim = len(shape_in)
            order = 3 if is_np else 1
            img = change_img_pixel_spacing(img, spacing[:idim],
                                           self.target_spacing[:idim],
                                           order=order)

        # downsampling
        if self.apply_downsampling:
            sp = self.target_spacing if self.apply_resizing else\
                spacing
            img = self._downsample_img(img, sp)

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
        is_seg = [False] + [True for _ in range(nch-1)]
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
        shape = np.array(volume.shape)
        if len(shape) == 3:
            if is_np:
                volume = volume[np.newaxis]
            else:
                volume = volume.unsqueeze(0)
        volume = self.preprocess_sample(volume, spacing)
        if len(shape) == 3:
            return volume[0]
        else:
            return volume

    def __call__(self, volume, spacing=None):
        return self.preprocess_volume(volume, spacing)

    def _find_nii_subfolder(self, nii_folder):
        subdirs = [d for d in my_listdir(nii_folder) if
                   isdir(join(nii_folder, d))]
        if 'labels' in subdirs:
            lbp = join(nii_folder, 'labels')
        elif 'labelsTr' in subdirs:
            lbp = join(nii_folder, 'labelsTr')
        else:
            raise FileNotFoundError('Didn\'t find label folder in '+nii_folder
                                    + '. Label folders are supposed to be '
                                    'named \'labels\' or \'labelsTr\'.')
        if 'images' in subdirs:
            imp = join(nii_folder, 'images')
        elif 'imagesTr' in subdirs:
            imp = join(nii_folder, 'imagesTr')
        else:
            raise FileNotFoundError('Didn\'t find image folder in '+nii_folder
                                    + '. Image folders are supposed to be '
                                    'named \'images\' or \'imagesTr\'.')
        return imp, lbp

    def preprocess_nii_folder(self, nii_folder, outfolder, use_torch=False):
        # first let's get the image and label path
        imp, lbp = self._find_nii_subfolder(nii_folder)
        # now let's create the output folders
        for f in ['images', 'labels', 'spacings', 'orig_shapes']:
            maybe_create_path(join(outfolder, f))
        # now let's get all the cases
        cases = [case[:-7] for case in my_listdir(lbp)
                 if case.endswith('.nii.gz')]

        # Let's quickly store the parameters so we can check later
        # what we've done here.
        self.save_attributes(outfolder)
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
            sample, spacing = read_nii_files(nii_files)
            sample[1] = self._remove_classes(sample[1])
            if self.use_only_fg_scans and sample[1].max() == 0:
                continue
            else:
                orig_shape = sample[0].shape
                if use_torch:
                    sample = torch.tensor(sample).to('cuda')
                sample = self.preprocess_sample(sample, spacing)
                if use_torch:
                    sample = sample.cpu().numpy()
                im, lb = sample[0].astype(np.float32),\
                    sample[1].astype(np.int8)
                for arr, folder in [[im, 'images'], [lb, 'labels'],
                                    [orig_shape, 'orig_shapes'],
                                    [spacing, 'spacings']]:
                    np.save(join(outfolder, folder, case), arr)
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

    def plan_preprocessing_from_nii(self, nii_folder, percentiles=[0.5, 99.5]):
        print('Infering preprocessing parameters from '+nii_folder)
        # first let's get the image and label path
        imp, lbp = self._find_nii_subfolder(nii_folder)
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
            print('Spacing: ({:.4f}, {:.4f}, {:.4f})'.
                  format(*self.target_spacing))
        if self.apply_windowing and self.window is None:
            self.window = np.percentile(fg_cvals, percentiles)
            print('Window: ({:.4f}, {:.4f})'.format(*self.window))
