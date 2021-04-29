import numpy as np
import torch
from torch.nn.functional import interpolate
from ovseg.utils.label_utils import remove_small_connected_components_from_batch, reduce_classes, \
    remove_small_connected_components
from ovseg.utils.dict_equal import dict_equal
from ovseg.utils.io import read_data_tpl_from_nii, load_pkl, save_pkl, save_txt
from ovseg.utils.path_utils import my_listdir, maybe_create_path
from ovseg.data.Dataset import raw_Dataset
from os.path import join, isdir, exists
from os import environ, listdir
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    print('No tqdm found, using no pretty progressing bars')
    tqdm = lambda x: x

from skimage.measure import block_reduce
from skimage.transform import rescale
from time import sleep


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
                 apply_resizing: bool,
                 apply_pooling: bool,
                 apply_windowing: bool,
                 target_spacing=None,
                 pooling_stride=None,
                 window=None,
                 scaling=None,
                 lb_classes=None,
                 reduce_lb_to_single_class=False,
                 lb_min_vol=None,
                 n_im_channels: int = 1,
                 do_nn_img_interp=False,
                 save_only_fg_scans=True,
                 dataset_properties={}):

        # first the parameters that determine the preprocessing operations
        self.apply_resizing = apply_resizing
        self.apply_pooling = apply_pooling
        self.apply_windowing = apply_windowing
        self.target_spacing = target_spacing
        self.pooling_stride = pooling_stride
        self.window = window
        self.scaling = scaling
        self.lb_classes = lb_classes
        self.reduce_lb_to_single_class = reduce_lb_to_single_class
        self.lb_min_vol = lb_min_vol
        self.n_im_channels = n_im_channels
        self.do_nn_img_interp = do_nn_img_interp
        # this is only important for preprocessing of raw data
        self.save_only_fg_scans = save_only_fg_scans
        self.dataset_properties = dataset_properties
        # when inheriting from this please append any other important
        # parameters that define the preprocessing
        self.preprocessing_parameters = ['apply_resizing',
                                         'apply_pooling',
                                         'apply_windowing',
                                         'target_spacing',
                                         'pooling_stride',
                                         'window',
                                         'scaling',
                                         'lb_classes',
                                         'reduce_lb_to_single_class',
                                         'lb_min_vol',
                                         'n_im_channels',
                                         'do_nn_img_interp',
                                         'save_only_fg_scans',
                                         'dataset_properties']

        self.is_initalised = False

        if self.check_parameters():
            self.initialise_preprocessing()
        else:
            # some parameters are missing
            print('Preprocessing was not initialized with necessary parameters. '
                  'Either load these with \'try_load_preprocessing_parameters\', '
                  'or infere them from raw data with \'plan_preprocessing_from_raw_data\'.'
                  'If you modify these parameters call \'initialise_preprocessing\'.')

    def check_parameters(self):

        if self.scaling is None:
            return False

        if self.apply_resizing and self.target_spacing is None:
            return False

        if self.apply_pooling and self.pooling_stride is None:
            return False

        if self.apply_windowing and self.window is None:
            return False

        return True

    def initialise_preprocessing(self):

        if not self.check_parameters():
            return

        inpt_dict_3d = {key: self.__getattribute__(key) for key in
                        self.preprocessing_parameters[:-2]}
        inpt_dict_2d = inpt_dict_3d.copy()
        if self.apply_resizing:
            inpt_dict_2d['target_spacing'] = self.target_spacing[1:]
        if self.apply_pooling:
            inpt_dict_2d['pooling_stride'] = self.pooling_stride[1:]
        inpt_dict_2d['is_2d'] = True

        self.torch_preprocessing = torch_preprocessing(**inpt_dict_3d)
        self.np_preprocessing = np_preprocessing(**inpt_dict_3d)

        self.torch_preprocessing_2d = torch_preprocessing(**inpt_dict_2d)
        self.np_preprocessing_2d = np_preprocessing(**inpt_dict_2d)

        self.is_initalised = True

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
            save_txt(data, outfile[:-4])

    def try_load_preprocessing_parameters(self, path_to_params):
        if not path_to_params.endswith('preprocessing_parameters.pkl'):
            path_to_params = join(path_to_params, 'preprocessing_parameters.pkl')

        if not exists(path_to_params):
            raise FileNotFoundError('No preprocessing parameters found at '+path_to_params)

        print('Loading preprocessing parameters from '+path_to_params)
        data = load_pkl(path_to_params)
        for key in data:
            self.__setattr__(key, data[key])
            print(str(key) + ': ' + str(data[key]))
        self.initialise_preprocessing()

    def maybe_clean_label_from_data_tpl(self, data_tpl):

        spacing = data_tpl['spacing'] if 'spacing' in data_tpl else None

        if 'label' not in data_tpl:
            raise ValueError('Can\'t clean label from data tpl, none was found!')

        lb = data_tpl['label']

        if self.is_preprocessed_data_tpl(data_tpl):
            return lb

        if self.lb_classes is not None:
            lb = reduce_classes(lb, self.lb_classes, self.reduce_lb_to_single_class)

        if self.lb_min_vol is not None:
            if len(lb.shape) > 3:
                lb = remove_small_connected_components_from_batch(lb, self.lb_min_vol, spacing)
            else:
                lb = remove_small_connected_components(lb, self.lb_min_vol, spacing)

        return lb

    def is_preprocessed_data_tpl(self, data_tpl):
        return 'orig_shape' in data_tpl

    def __call__(self, data_tpl, preprocess_only_im=False, return_np=False):

        spacing = data_tpl['spacing'] if 'spacing' in data_tpl else None
        if 'image' not in data_tpl:
            raise ValueError('No \'image\' found in data_tpl')
        xb = data_tpl['image']

        assert len(xb.shape) in [3, 4], 'image must be 3d or 4d'
        if len(xb.shape) == 3:
            xb = xb[np.newaxis]

        if 'label' in data_tpl and not preprocess_only_im:
            lb = data_tpl['label']

            assert len(lb.shape) == 3, 'label must be 3d'
            lb = lb[np.newaxis]
            xb = np.concatenate([xb, lb])

        xb = xb[np.newaxis]
        # now do the preprocessing
        if not torch.cuda.is_available():
            # the preprocessing is also faster in scipy then it is in torch using the CPU
            xb_prep = self.np_preprocessing(xb, spacing)
        else:
            xb_cuda = torch.from_numpy(xb).type(torch.float).cuda()
            try:
                xb_prep = self.torch_preprocessing(xb_cuda, spacing)
                if return_np:
                    xb_prep = xb_prep.cpu().numpy()
            except RuntimeError:
                print('Ooops! It seems like your GPU has gone out of memory while trying to '
                      'resize a large volume ({}), trying again on the CPU.'
                      ''.format(list(xb_cuda.shape)))
                torch.cuda.empty_cache()
                xb_prep = self.np_preprocessing(xb, spacing)
                if not return_np:
                    xb_prep = torch.from_numpy(xb_prep).type(torch.float).cuda()

        return xb_prep[0]

    def preprocess_raw_data(self,
                            raw_data,
                            preprocessed_name='default',
                            data_name=None,
                            save_as_fp16=True,
                            image_folder=None,
                            dcm_revers=True,
                            dcm_names_dict=None):

        if isinstance(raw_data, str):
            raw_data = [raw_data]
        elif not isinstance(raw_data, (tuple, list)):
            raise ValueError('raw_data must be str if only infered from a sinlge folder or '
                             'list/tuple.')

        if not self.is_initalised:
            print('Preprocessing classes were not initialised when calling '
                  '\'preprocess_raw_data\'. Doing it now.\n')
            self.initialise_preprocessing()

        im_dtype = np.float16 if save_as_fp16 else np.float32

        if data_name is None:
            data_name = '_'.join(sorted(raw_data))

        # root folder of all saved preprocessed data
        outfolder = join(environ['OV_DATA_BASE'], 'preprocessed', data_name, preprocessed_name)
        plot_folder = join(environ['OV_DATA_BASE'], 'plots', data_name, preprocessed_name)
        print(outfolder, plot_folder)
        # now let's create the output folders
        for f in ['images', 'labels', 'fingerprints']:
            maybe_create_path(join(outfolder, f))
        maybe_create_path(plot_folder)

        # Let's quickly store the parameters so we can check later
        # what we've done here.
        self.maybe_save_preprocessing_parameters(outfolder)
        # here is the fun
        print()
        for raw_name in raw_data:
            print('Converting ' + raw_name)
            raw_ds = raw_Dataset(join(environ['OV_DATA_BASE'], 'raw_data', raw_name),
                                 image_folder=image_folder,
                                 dcm_revers=dcm_revers,
                                 dcm_names_dict=dcm_names_dict)
            print()
            sleep(1)
            for i in tqdm(range(len(raw_ds))):
                # read files
                data_tpl = raw_ds[i]

                im, spacing = data_tpl['image'], data_tpl['spacing']

                orig_shape = im.shape[-3:]
                orig_spacing = spacing.copy()
                if 'label' not in data_tpl:
                    data_tpl['label'] = np.zeros(orig_shape)
                xb = self.__call__(data_tpl, return_np=True)
                im = xb[:self.n_im_channels].astype(im_dtype)
                lb = xb[self.n_im_channels:].astype(np.uint8)
                if lb.max() == 0 and self.save_only_fg_scans:
                    continue
                spacing = self.target_spacing if self.apply_resizing else spacing
                if self.apply_pooling:
                    spacing = np.array(spacing) * np.array(self.pooling_stride)
                fingerprint_keys = [key for key in data_tpl if key not in ['image', 'label']]
                fingerprint = {key: data_tpl[key] for key in fingerprint_keys}
                fingerprint['orig_shape'] = orig_shape
                fingerprint['orig_spacing'] = orig_spacing
                fingerprint['spacing'] = spacing
                scan = data_tpl['scan']
                if 'dataset' not in fingerprint:
                    fingerprint['dataset'] = raw_name
                if 'pat_id' not in fingerprint:
                    fingerprint['pat_id'] = scan
                # first save the image related stuff
                for arr, folder in [[np.squeeze(im, 0), 'images'],
                                    [np.squeeze(lb, 0), 'labels'],
                                    [fingerprint, 'fingerprints']]:
                    np.save(join(outfolder, folder, scan), arr)

                # additionally do some plots
                lb = np.sum(lb, 0) > 0
                im = im.astype(float)

                contains = np.where(np.sum(lb, (1, 2)))[0]
                z_list = [np.argmax(np.sum(lb, (1, 2)))]
                s_list = ['_largest', '_random']
                if len(contains) > 0:
                    z_list.extend(np.random.choice(contains, size=1))
                else:
                    z_list.extend(np.random.randint(lb.shape[0], size=1))
                n_ch = im.shape[0]
                for z, s in zip(z_list, s_list):
                    fig = plt.figure()
                    for c in range(n_ch):
                        plt.subplot(1, n_ch, c+1)
                        plt.imshow(im[c, z], cmap='gray')
                        if lb[z].max() > 0:
                            # this if is purely to avoid annoying UserWarning messages that
                            # interrupt the beautiful beautiful tqdm bar
                            plt.contour(lb[z] > 0, linewidths=0.5, colors='red',
                                        linestyles='dashed')
                        plt.axis('off')
                    plt.savefig(join(plot_folder, scan + s + '.png'))
                    plt.close(fig)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print('Preprocessing done!')

    def plan_preprocessing_raw_data(self, raw_data,
                                    percentiles=[0.5, 99.5],
                                    image_folder=None,
                                    dcm_revers=True,
                                    dcm_names_dict=None):

        if isinstance(raw_data, str):
            raw_data = [raw_data]
        elif not isinstance(raw_data, (tuple, list)):
            raise ValueError('raw_data must be str if only infered from a sinlge folder or '
                             'list/tuple.')

        if self.check_parameters():
            print('It seems like all necessary information is given. Skipping the planning!\n\n')
            return

        # first let's get the image and label path
        print('Infering preprocessing parameters from ', *raw_data)
        print('Creating datasets...')
        datasets = []
        for data_name in raw_data:
            print('Reading ' + data_name)
            raw_ds = raw_Dataset(join(environ['OV_DATA_BASE'], 'raw_data', data_name),
                                 image_folder=image_folder,
                                 dcm_revers=dcm_revers,
                                 dcm_names_dict=dcm_names_dict)
            datasets.append(raw_ds)
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
        for raw_name, raw_ds in zip(raw_data, datasets):
            print(raw_name)
            print()
            # Yes I am wasting 1 sec of your time to ensure that the tqdm bars are not
            # interupted ;)
            sleep(1)
            for i in tqdm(range(len(raw_ds))):
                # read files
                data_tpl = raw_ds[i]

                im, spacing = data_tpl['image'], data_tpl['spacing']
                lb = self.maybe_clean_label_from_data_tpl(data_tpl)
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
            fg_cvals = np.array(fg_cvals)#.clip(*fg_percentiles)
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
        n_cases = 0
        print()
        print('Second cycle')
        print()
        for raw_name, raw_ds in zip(raw_data, datasets):
            print(raw_name)
            print()
            n_cases += len(raw_ds)
            sleep(1)
            for i in tqdm(range(len(raw_ds))):
                # read files
                data_tpl = raw_ds[i]

                im, lb, spacing = data_tpl['image'], data_tpl['label'], data_tpl['spacing']
                im = im.astype(float)
                im_win = im.clip(*self.window)
                mean_global += np.mean(im)
                mean_window += np.mean(im_win)
                mean2_global += np.mean(im**2)
                mean2_window += np.mean(im_win**2)
        mean_global, mean2_global = mean_global/n_cases, mean2_global/n_cases
        mean_window, mean2_window = mean_window/n_cases, mean2_window/n_cases
        std_global = np.sqrt(mean2_global - mean_global**2)
        std_window = np.sqrt(mean2_window - mean_window**2)
        self.dataset_properties['scaling_global'] = \
            np.array([std_global, mean_global]).astype(np.float32)
        self.dataset_properties['scaling_window'] = \
            np.array([std_window, mean_window]).astype(np.float32)
        print('Done!\n')
        if self.scaling is None:
            if self.apply_windowing:
                self.scaling = self.dataset_properties['scaling_foreground']
            else:
                self.scaling = self.dataset_properties['scaling_global']
            print('Scaling: ({:.4f}, {:.4f})'.format(*self.scaling))
        if self.apply_resizing:
            print('Spacing: ({:.4f}, {:.4f}, {:.4f})'.format(*self.target_spacing))
        if self.apply_windowing:
            print('Window: ({:.4f}, {:.4f})'.format(*self.window))
        print()


# %% Let's be fancy and do the preprocessing for np and torch as seperate operators
class torch_preprocessing(torch.nn.Module):

    # preprocessing module for 2d and 3d

    def __init__(self,
                 apply_resizing: bool,
                 apply_pooling: bool,
                 apply_windowing: bool,
                 target_spacing=None,
                 pooling_stride=None,
                 window=None,
                 scaling=[1, 0],
                 lb_classes=None,
                 reduce_lb_to_single_class=False,
                 lb_min_vol=None,
                 n_im_channels: int = 1,
                 do_nn_img_interp=False,
                 is_2d=False):
        super().__init__()
        self.apply_resizing = apply_resizing
        self.apply_pooling = apply_pooling
        self.apply_windowing = apply_windowing
        self.n_im_channels = n_im_channels
        self.lb_classes = lb_classes
        self.reduce_lb_to_single_class = reduce_lb_to_single_class
        self.lb_min_vol = lb_min_vol
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.is_2d = is_2d
        self.do_nn_img_interp = do_nn_img_interp

        # let's test if the inputs were fine
        if self.apply_resizing:
            self.target_spacing = np.array(target_spacing)
            if self.is_2d:
                assert len(target_spacing) == 2, 'target spacing must be of length 2'
                self.mode = 'bilinear'
            else:
                assert len(target_spacing) == 3, 'target spacing must be of length 3'
                self.mode = 'trilinear'

        if self.do_nn_img_interp:
            self.mode = 'nearest'

        if self.apply_pooling:
            if self.is_2d:
                assert len(pooling_stride) == 2, 'pooling stride must be of length 3'
                self.pooling_stride = pooling_stride
                self.mean_pooling = torch.nn.AvgPool2d(kernel_size=self.pooling_stride,
                                                       stride=self.pooling_stride)
                self.max_pooling = torch.nn.MaxPool2d(kernel_size=self.pooling_stride,
                                                      stride=self.pooling_stride)
            else:
                assert len(pooling_stride) == 3, 'pooling stride must be of length 3'
                self.pooling_stride = pooling_stride
                self.mean_pooling = torch.nn.AvgPool3d(kernel_size=self.pooling_stride,
                                                       stride=self.pooling_stride)
                self.max_pooling = torch.nn.MaxPool3d(kernel_size=self.pooling_stride,
                                                      stride=self.pooling_stride)

        if self.apply_windowing:
            assert len(window) == 2, 'window must be of length 2'
            self.window = window

        assert len(scaling) == 2, 'scaling must be of length 2 (std, mean)'
        self.scaling = scaling

    def forward(self, xb, spacing=None):

        # assume the image channels are always first
        n_ch = xb.shape[1]
        imb = xb[:, :self.n_im_channels]
        has_lb = n_ch > self.n_im_channels
        if has_lb:
            lbb = xb[:, self.n_im_channels:]

            if self.lb_classes is not None and self.lb_min_vol is not None:
                lbb = lbb.cpu().numpy()
                lbb = reduce_classes(lbb, self.lb_classes, self.reduce_lb_to_single_class)
                lbb = remove_small_connected_components_from_batch(lbb, self.lb_min_vol, spacing)
                lbb = torch.from_numpy(lbb).to(self.dev)

            elif self.lb_classes is not None:
                lbb = lbb.cpu().numpy()
                lbb = reduce_classes(lbb, self.lb_classes, self.reduce_lb_to_single_class)
                lbb = torch.from_numpy(lbb).to(self.dev)

            elif self.lb_min_vol is not None:
                lbb = lbb.cpu().numpy()
                lbb = remove_small_connected_components_from_batch(lbb, self.lb_min_vol, spacing)
                lbb = torch.from_numpy(lbb).to(self.dev)

        # resizing
        if self.apply_resizing:

            scale_factor = (spacing / self.target_spacing).tolist()

            imb = interpolate(imb, scale_factor=scale_factor, mode=self.mode)
            if has_lb:
                lbb = interpolate(lbb, scale_factor=scale_factor)

        # pooling
        if self.apply_pooling:

            imb = self.mean_pooling(imb)
            if has_lb:
                lbb = self.max_pooling(lbb)

        # windowing:
        if self.apply_windowing:

            imb = imb.clamp(*self.window)

        # scaling
        imb = (imb - self.scaling[1]) / self.scaling[0]

        if has_lb:
            xb = torch.cat([imb, lbb], 1)
        else:
            xb = imb

        return xb


# %%
class np_preprocessing():

    # preprocessing class for 2d and 3d np niput

    def __init__(self,
                 apply_resizing: bool,
                 apply_pooling: bool,
                 apply_windowing: bool,
                 target_spacing=None,
                 pooling_stride=None,
                 window=None,
                 scaling=[1, 0],
                 lb_classes=None,
                 reduce_lb_to_single_class=False,
                 lb_min_vol=None,
                 n_im_channels: int = 1,
                 do_nn_img_interp=False,
                 is_2d=False):
        super().__init__()
        self.apply_resizing = apply_resizing
        self.apply_pooling = apply_pooling
        self.apply_windowing = apply_windowing
        self.n_im_channels = n_im_channels
        self.lb_classes = lb_classes
        self.reduce_lb_to_single_class = reduce_lb_to_single_class
        self.lb_min_vol = lb_min_vol
        self.is_2d = is_2d
        self.do_nn_img_interp = do_nn_img_interp
        self.img_order = 0 if self.do_nn_img_interp else 1

        # let's test if the inputs were fine
        if self.apply_resizing:
            self.target_spacing = np.array(target_spacing)
            if self.is_2d:
                assert len(target_spacing) == 2, 'target spacing must be of length 2'
            else:
                assert len(target_spacing) == 3, 'target spacing must be of length 3'

        if self.apply_pooling:
            self.pooling_stride = pooling_stride
            if self.is_2d:
                assert len(pooling_stride) == 2, 'pooling stride must be of length 3'
            else:
                assert len(pooling_stride) == 3, 'pooling stride must be of length 3'

        if self.apply_windowing:
            assert len(window) == 2, 'window must be of length 2'
            self.window = window

        assert len(scaling) == 2, 'scaling must be of length 2 (std, mean)'
        self.scaling = scaling

    def maybe_clean_label(self, lb, spacing=None):

        if self.lb_classes is not None:
            lb = reduce_classes(lb, self.lb_classes, self.reduce_lb_to_single_class)

        if self.lb_min_vol is not None:
            if len(lb.shape) > 3:
                lb = remove_small_connected_components_from_batch(lb, self.lb_min_vol, spacing)
            else:
                lb = remove_small_connected_components(lb, self.lb_min_vol, spacing)

        return lb

    def _rescale_batch(self, im, spacing, order=1):

        if spacing is None:
            raise ValueError('spacing must be given as input when apply_resizing=True.')

        bs, nch = im.shape[0:2]
        idim = 2 if self.is_2d else 3
        scale = np.array(spacing) / self.target_spacing
        shape = im.shape[-1*idim:]
        im_vec = im.reshape(-1, *shape)
        im_vec = np.stack([rescale(im_vec[i], scale, order=order) for i in range(im_vec.shape[0])])
        return im_vec.reshape(bs, nch, *im_vec.shape[1:])

    def __call__(self, xb, spacing=None):

        inpt_dim = 4 if self.is_2d else 5
        assert len(xb.shape) == inpt_dim, 'input images must be {}d tensor'.format(inpt_dim)

        # assume the image channels are always first
        n_ch = xb.shape[1]
        imb = xb[:, :self.n_im_channels]
        has_lb = n_ch > self.n_im_channels
        if has_lb:
            lbb = xb[:, self.n_im_channels:]

            lbb = self.maybe_clean_label(lbb, spacing)

        # resizing
        if self.apply_resizing:

            imb = self._rescale_batch(imb, spacing, order=self.img_order)
            if has_lb:
                lbb = self._rescale_batch(lbb, spacing, order=0)

        # pooling
        if self.apply_pooling:

            imb = block_reduce(imb, (1, 1, *self.pooling_stride), func=np.mean)
            if has_lb:
                lbb = block_reduce(lbb, (1, 1, *self.pooling_stride), func=np.max)

        # windowing:
        if self.apply_windowing:

            imb = imb.clip(*self.window)

        # scaling
        imb = (imb - self.scaling[1]) / self.scaling[0]

        if has_lb:
            xb = np.concatenate([imb, lbb], 1)
        else:
            xb = imb

        return xb
