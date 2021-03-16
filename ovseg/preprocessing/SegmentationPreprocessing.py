import numpy as np
import torch
from ovseg.utils.interp_utils import change_img_pixel_spacing,\
    change_sample_pixel_spacing
from ovseg.utils.torch_np_utils import check_type, stack
from ovseg.utils.io import read_data_tpl_from_nii, load_pkl, save_pkl
from ovseg.utils.path_utils import my_listdir, maybe_create_path
from ovseg.utils.dict_equal import dict_equal
from os.path import join, isdir, exists, basename
from os import environ, listdir
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    print('No tqdm found, using no pretty progressing bars')
    tqdm = lambda x: x


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
                 normalise='window',
                 apply_downsampling=False,
                 downsampling_fac=None,
                 use_only_fg_scans=True,
                 use_only_classes=None,
                 reduce_to_single_class=False,
                 try_preprocess_volumes_on_gpu=True,
                 label_interpolation='nearest',
                 **kwargs):

        self.apply_resizing = apply_resizing
        self.target_spacing = target_spacing
        self.apply_windowing = apply_windowing
        self.window = window
        self.apply_scaling = apply_scaling
        self.scaling = scaling
        assert normalise in ['foreground', 'window', 'global']
        self.normalise = normalise
        self.apply_downsampling = apply_downsampling
        self.downsampling_fac = downsampling_fac
        self.use_only_fg_scans = use_only_fg_scans
        self.use_only_classes = use_only_classes
        self.reduce_to_single_class = reduce_to_single_class
        self.try_preprocess_volumes_on_gpu = try_preprocess_volumes_on_gpu
        assert label_interpolation in ['nearest', 'nnUNet']
        self.label_interpolation = label_interpolation
        self.force_volume_preprocessing_to_cpu = False
        self.dataset_properties = {}

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
                                         'reduce_to_single_class',
                                         'label_interpolation',
                                         'use_only_fg_scans',
                                         'dataset_properties',
                                         'normalise']

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

    def maybe_save_preprocessing_parameters(self, outfolder):
        outfile = join(outfolder, 'preprocessing_parameters.pkl')
        data = {key: self.__getattribute__(key) for key in
                self.preprocessing_parameters}
        if exists(outfile):
            data_pkl = load_pkl(outfile)
            if dict_equal(data_pkl, data):
                return
            else:
                raise RuntimeError('Found not matching prerpocessing parameters in '+outfolder+'.')
        else:
            save_pkl(data, outfile)

    def load_preprocessing_parameters(self, path_to_params):
        if not path_to_params.endswith('preprocessing_parameters.pkl'):
            path_to_params = join(path_to_params, 'preprocessing_parameters.pkl')
        print('Loading preprocessing parameters from '+path_to_params)
        data = load_pkl(path_to_params)
        for key in data:
            self.__setattr__(key, data[key])

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

            # nearest neighbour interpolation is easy! Plus it should be very similar to
            # what nnUNet does
            if self.label_interpolation == 'nearest':

                idim = len(shape_in)
                return change_img_pixel_spacing(img, spacing[:idim],
                                                self.target_spacing[:idim],
                                                order=0)
            else:
                # this is what nnUNet does
                # convert to one hot encoding
                dtype = img.dtype
                nclasses = int(img.max())
                img = stack([img == c for c in range(nclasses+1)])
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
                    img = np.argmax(img, 0).astype(dtype)
                else:
                    img = torch.argmax(img, 0).type(dtype)
                return img

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

    def preprocess_sample(self, sample, spacing=None, is_seg=None):
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
        if is_seg is None:
            is_seg = [False] + [True for _ in range(nch-1)]
        elif isinstance(is_seg, bool):
            is_seg = nch * [is_seg]
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

    def preprocess_volume(self, volume, spacing=None, is_seg=None):
        '''
        Preprocesses a full volume.

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
        shape_inpt = np.array(volume.shape)
        if len(shape_inpt) == 3:
            if is_np:
                volume = volume[np.newaxis]
            else:
                volume = volume.unsqueeze(0)

        if not is_np and self.force_volume_preprocessing_to_cpu:
            dev = volume.device
            volume = volume.cpu().numpy()

        # now let's check if we're using the GPU
        use_gpu = self.try_preprocess_volumes_on_gpu and torch.cuda.device_count() > 0 \
            and not self.force_volume_preprocessing_to_cpu

        if use_gpu:
            if is_np:
                volume = torch.from_numpy(volume)
            volume = volume.type(torch.float16).cuda()

        try:
            with torch.no_grad():
                volume = self.preprocess_sample(volume, spacing, is_seg).type(torch.float32)
        except RuntimeError:
            print('RuntimeError caught! This is most likely due to a cuda oom error when '
                  'resizing a large volume. Trying again on CPU.')
            volume = self.preprocess_sample(volume.cpu().numpy(), spacing)
            volume = torch.from_numpy(volume)
            torch.cuda.empty_cache()

        if use_gpu and is_np:
            # we're a bit lazy here
            # if the input is a torch cpu tensor we're not transferring back
            volume = volume.cpu().numpy()

        if not is_np and self.force_volume_preprocessing_to_cpu:
            volume = torch.from_numpy(volume).to(dev)

        if len(shape_inpt) == 3:
            volume = volume[0]

        return volume

    def preprocess_volume_from_data_tpl(self, data_tpl, return_seg):
        if 'orig_shape' in data_tpl:
            # skip! This data_tpl is already preprocessed
            if return_seg:
                return data_tpl['label']
            else:
                return data_tpl['image']
        else:
            # data_tpl is not preprocessed. Let's do some work
            if return_seg:
                # in this case we only want to remove the labels and not do any resizing or such
                return self._maybe_reduce_label(data_tpl['label'])
            else:
                return self.preprocess_volume(data_tpl['image'], data_tpl['spacing'])

    def __call__(self, volume, spacing=None, is_seg=None):
        return self.preprocess_volume(volume, spacing, is_seg=None)

    def _maybe_reduce_label(self, lb):

        # remove classes from the label for the case we don't want to segment
        if isinstance(self.use_only_classes, (list, tuple)):
            lb_new = np.zeros_like(lb)
            for i, c in enumerate(self.use_only_classes):
                lb_new[lb == c] = i+1
            lb = lb_new

        # in case we want to do only differentiate fg and bg (abnormality segmentation)
        if self.reduce_to_single_class:
            lb = (lb > 0).astype(lb.dtype)

        return lb

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

    def _get_all_cases_from_raw_data(self, raw_data):
        if isinstance(raw_data, str):
            raw_data = [raw_data]
        elif not isinstance(raw_data, (tuple, list)):
            raise ValueError('raw_data must be str if only infered from a sinlge folder or '
                             'list/tuple.')

        raw_data_name = '_'.join(raw_data)
        # check for exsistence and collect cases
        folders_and_cases = []

        raw_data_path = join(environ['OV_DATA_BASE'], 'raw_data')
        for data_fol in raw_data:
            data_path = join(raw_data_path, data_fol)
            if not exists(data_path):
                raise FileNotFoundError('Did not find path '+data_path)

            imp, lbp = self._find_nii_subfolder(data_path)
            for case in listdir(lbp):
                name = case[:-7]
                im_cases = [case for case in listdir(imp) if case.startswith(name)]
                if len(im_cases) == 0:
                    print('Found no image file for label file {}.'.format(case))
                    continue
                folders_and_cases.append((data_fol, case))

        return folders_and_cases, raw_data_name

    def preprocess_raw_data(self, raw_data, data_name=None, preprocessed_name='default',
                            save_as_fp16=False):

        im_dtype = np.float16 if save_as_fp16 else np.float32

        # get cases and name
        folders_and_cases, raw_data_name = self._get_all_cases_from_raw_data(raw_data)

        if data_name is None:
            data_name = raw_data_name

        # root folder of all saved preprocessed data
        outfolder = join(environ['OV_DATA_BASE'], 'preprocessed', data_name, preprocessed_name)
        plot_folder = join(environ['OV_DATA_BASE'], 'plots', data_name, preprocessed_name)
        # now let's create the output folders
        for f in ['images', 'labels', 'fingerprints']:
            maybe_create_path(join(outfolder, f))
        maybe_create_path(plot_folder)

        # Let's quickly store the parameters so we can check later
        # what we've done here.
        self.maybe_save_preprocessing_parameters(outfolder)
        # here is the fun
        print()
        for folder, case in tqdm(folders_and_cases):
            # read files
            data_tpl = read_data_tpl_from_nii(folder, case)

            im, lb, spacing = data_tpl['image'], data_tpl['label'], data_tpl['spacing']

            orig_shape = im.shape[-3:]
            orig_spacing = spacing.copy()
            # TODO allow more complex change of spacing here in case of downsampling or such
            im = self.preprocess_volume(im, spacing, is_seg=False).astype(im_dtype)
            lb = self.preprocess_volume(lb, spacing, is_seg=True).astype(np.int8)
            lb = self._maybe_reduce_label(lb)
            if lb.max() == 0 and self.use_only_fg_scans:
                continue
            spacing = self.target_spacing if self.apply_resizing else spacing
            fingerprint_keys = [key for key in data_tpl if key not in ['image', 'label']]
            fingerprint = {key: data_tpl[key] for key in fingerprint_keys}
            fingerprint['orig_shape'] = orig_shape
            fingerprint['orig_spacing'] = orig_spacing
            fingerprint['spacing'] = spacing
            if 'dataset' not in fingerprint:
                fingerprint['dataset'] = folder
            if 'pat_id' not in fingerprint:
                fingerprint['pat_id'] = case.split('.')[0]
            # first save the image related stuff
            name = case[:-7]
            for arr, folder in [[im, 'images'],
                                [lb, 'labels'],
                                [fingerprint, 'fingerprints']]:
                np.save(join(outfolder, folder, name), arr)

            # additionally do some plots
            if len(lb.shape) != 3:
                continue
            if len(im.shape) == 3:
                im = im[np.newaxis]

            contains = np.where(np.sum(lb, (0, 1)))[0]
            z_list = [np.argmax(np.sum(lb, (0, 1)))]
            s_list = ['_largest', '_random_0', '_random_1', '_random_2']
            z_list.extend(np.random.choice(contains, size=3))
            n_ch = im.shape[0]
            for z, s in zip(z_list, s_list):
                fig = plt.figure()
                for c in range(n_ch):
                    plt.subplot(1, n_ch, c+1)
                    plt.imshow(im[c, ..., z], cmap='gray')
                    if lb[..., z].max() > 0:
                        # this if is purely to avoid annoying UserWarning messages that interrupt
                        # the beautiful beautiful tqdm bar
                        plt.contour(lb[..., z] > 0, linewidths=0.5, colors='red',
                                    linestyles='dashed')
                    plt.axis('off')
                plt.savefig(join(plot_folder, name + s + '.png'))
                plt.close(fig)

        torch.cuda.empty_cache()
        print('Preprocessing done!')

    def plan_preprocessing_raw_data(self, raw_data, percentiles=[0.5, 99.5]):

        # first let's get the image and label path
        folders_and_cases, raw_data_name = self._get_all_cases_from_raw_data(raw_data)
        print('Infering preprocessing parameters from '+raw_data_name)
        # foreground vals for windowing
        fg_cvals = []
        # spacings for resampling
        spacings = []
        shapes = []
        n_fg_classes = 0
        # first cycle
        print()
        print('First cycle')
        print()
        for folder, case in tqdm(folders_and_cases):
            # read files
            data_tpl = read_data_tpl_from_nii(folder, case)

            im, lb, spacing = data_tpl['image'], data_tpl['label'], data_tpl['spacing']
            lb = self._maybe_reduce_label(lb)
            # store spacing
            spacings.append(spacing)
            shapes.append(im.shape)
            if lb.max() > 0:
                fg_cval = im[lb > 0].astype(float)
                fg_cvals.extend(fg_cval.tolist())
                n_fg_classes = np.max([n_fg_classes, lb.max()])

        if len(fg_cvals) > 10**8:
            # in some datasets the length of fg_cvals can become too long
            # in this case we split up the lists into chunks of data that can be put into an
            # np array and we compute the mean over the statistics of each batch
            fg_percentile_list = []
            n_arrays = len(fg_cvals) // 10 ** 8 + 1
            array_len = len(fg_cvals) // n_arrays
            for i in range(n_arrays - 1):
                fg_percentile_list.append(np.percentile(fg_cvals[i * array_len:
                                                                 (i+1) * array_len],
                                                        percentiles))
            fg_percentile_list.append(np.percentile(fg_cvals[(n_arrays - 1) * array_len:],
                                                    percentiles))
            fg_percentiles = np.mean(fg_percentile_list, 0)
            std_fg_list, mean_fg_list = [], []
            for i in range(n_arrays - 1):
                std_fg_list.append(np.std(np.clip(fg_cvals[i * array_len: (i+1) * array_len],
                                                  *fg_percentiles)))
                mean_fg_list.append(np.mean(np.clip(fg_cvals[i * array_len: (i+1) * array_len],
                                                    *fg_percentiles)))
            std_fg_list.append(np.std(fg_cvals[(n_arrays - 1) * array_len:]))
            mean_fg_list.append(np.mean(fg_cvals[(n_arrays - 1) * array_len:]))
            std_fg, mean_fg = np.mean(std_fg_list), np.mean(mean_fg_list)
        else:
            fg_percentiles = np.percentile(fg_cvals, percentiles)
            fg_cvals = np.array(fg_cvals).clip(*fg_percentiles)
            std_fg, mean_fg = np.std(fg_cvals), np.mean(fg_cvals)

        self.dataset_properties['median_shape'] = np.median(shapes, 0)
        self.dataset_properties['median_spacing'] = np.median(spacings, 0)
        self.dataset_properties['fg_percentiles'] = fg_percentiles
        self.dataset_properties['percentiles'] = percentiles
        self.dataset_properties['scaling_foreground'] = \
            np.array([std_fg, mean_fg]).astype(np.float32)
        self.dataset_properties['n_fg_classes'] = n_fg_classes

        if self.apply_resizing and self.target_spacing is None:
            self.target_spacing = self.dataset_properties['median_spacing']
        if self.apply_windowing and self.window is None:
            self.window = self.dataset_properties['fg_percentiles']

        mean_global = 0
        mean_window = 0
        mean2_global = 0
        mean2_window = 0
        print()
        print('Second cycle')
        print()
        for folder, case in tqdm(folders_and_cases):
            # read files
            data_tpl = read_data_tpl_from_nii(folder, case)

            im, lb, spacing = data_tpl['image'], data_tpl['label'], data_tpl['spacing']
            im_win = im.clip(*self.window)
            mean_global += np.mean(im)
            mean_window += np.mean(im_win)
            mean2_global += np.mean(im**2)
            mean2_window += np.mean(im_win**2)
        n_cases = len(folders_and_cases)
        mean_global, mean2_global = mean_global/n_cases, mean2_global/n_cases
        mean_window, mean2_window = mean_window/n_cases, mean2_window/n_cases
        std_global = np.sqrt(mean2_global - mean_global**2)
        std_window = np.sqrt(mean2_window - mean_window**2)
        self.dataset_properties['scaling_global'] = \
            np.array([std_global, mean_global]).astype(np.float32)
        self.dataset_properties['scaling_window'] = \
            np.array([std_window, mean_window]).astype(np.float32)
        print('Done')
        if self.apply_scaling and self.scaling is None:
            if self.normalise == 'global':
                self.scaling = self.dataset_properties['scaling_global']
            elif self.normalise == 'window':
                self.scaling = self.dataset_properties['scaling_window']
            elif self.normalise == 'foreground':
                self.scaling = self.dataset_properties['scaling_foreground']
        if self.apply_scaling:
            print('Scaling: ({:.4f}, {:.4f})'.format(*self.scaling))
        if self.apply_resizing:
            print('Spacing: ({:.4f}, {:.4f}, {:.4f})'.format(*self.target_spacing))
        if self.apply_windowing:
            print('Window: ({:.4f}, {:.4f})'.format(*self.window))
