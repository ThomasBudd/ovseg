import numpy as np
import torch
from torch.nn.functional import interpolate
from ovseg.utils.label_utils import remove_small_connected_components_from_batch, reduce_classes, \
    remove_small_connected_components
from ovseg.utils.label_utils import remove_connected_components_by_volume_from_batch, \
    remove_connected_components_by_volume
from ovseg.utils.dict_equal import dict_equal, print_dict_diff
from ovseg.utils.io import load_pkl, save_pkl, save_txt
from ovseg.utils.path_utils import maybe_create_path
from ovseg.data.Dataset import raw_Dataset
from ovseg.utils.torch_np_utils import maybe_add_channel_dim
from ovseg import OV_PREPROCESSED
from os.path import join, exists
from os import environ
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    print('No tqdm found, using no pretty progressing bars')
    tqdm = lambda x: x

from skimage.measure import block_reduce
from skimage.transform import rescale
from time import sleep


class SegmentationPreprocessingV2(object):
    '''
    Class that is responsible for performing preprocessing of segmentation data
    This class expects
        1) single channel images
        2) non overlappting segmentation in integer encoding
    If the corresponding flags are set we perform
         1) resizing to change the pixel spacing to target_spacing
         2) additional downsampling
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
                 n_im_channels: int = 1,
                 do_nn_img_interp=False,
                 save_only_fg_scans=True,
                 prev_stage_for_input:dict={},
                 prev_stage_for_mask:dict={},
                 r_dial_mask=0,
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
        self.n_im_channels = n_im_channels
        self.do_nn_img_interp = do_nn_img_interp
        self.prev_stage_for_input = prev_stage_for_input
        self.prev_stage_for_mask = prev_stage_for_mask
        self.r_dial_mask = r_dial_mask
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
                                         'n_im_channels',
                                         'do_nn_img_interp',
                                         'save_only_fg_scans',
                                         'prev_stage_for_input',
                                         'prev_stage_for_mask',
                                         'r_dial_mask',
                                         'dataset_properties']
        
        self.has_ps_input = len(self.prev_stage_for_input) > 0
        self.has_ps_mask = len(self.prev_stage_for_mask) > 0
        
        # create keys for previous stages
        if self.has_ps_input:
            for key in ['data_name', 'preprocessed_name', 'model_name']:
                assert key in self.prev_stage_for_input
            self.key_ps_input = '_'.join(['prediction',
                                          self.prev_stage_for_input['data_name'],
                                          self.prev_stage_for_input['preprocessed_name'],
                                          self.prev_stage_for_input['model_name']])
        if self.has_ps_mask:
            for key in ['data_name', 'preprocessed_name', 'model_name']:
                assert key in self.prev_stage_for_mask
            self.key_ps_mask = '_'.join(['prediction',
                                         self.prev_stage_for_mask['data_name'],
                                         self.prev_stage_for_mask['preprocessed_name'],
                                         self.prev_stage_for_mask['model_name']])

        # creat previous stages
        if self.has_ps_input or self.has_ps_mask:
            self.prev_stages = []
            if self.has_ps_input:
                self.prev_stages.append(self.prev_stage_for_input)
            if self.has_ps_mask:
                self.prev_stages.append(self.prev_stage_for_mask)
        
            #self.prev_stages = list(set(self.prev_stages))
        else:
            self.prev_stages = None

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
            print('Not all required parameters were initialised, can not initialise '
                  'preprocessing objects')
            return

        include_keys = ['apply_resizing', 'apply_pooling', 'apply_windowing', 'target_spacing', 
                        'pooling_stride', 'window', 'scaling', 'n_im_channels',
                        'do_nn_img_interp']

        inpt_dict_3d = {key: self.__getattribute__(key) for key in
                        self.preprocessing_parameters if key in include_keys}
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
                print_dict_diff(data_pkl, data, 'pkl paramters', 'given paramters')
                print('Found some not matching prerpocessing parameters in '+outfolder+'.'
                      'The currently used parameters will not overwrite the stored ones.')
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

        if 'label' not in data_tpl:
            
            return np.zeros(data_tpl['image'].shape[-3:])

        lb = data_tpl['label']

        if self.is_preprocessed_data_tpl(data_tpl):
            return lb

        if self.lb_classes is not None:
            lb = reduce_classes(lb, self.lb_classes, self.reduce_lb_to_single_class)
        elif self.reduce_lb_to_single_class:
            # we we're not using certain classes we do the reduction
            # to binary label here
            lb = (lb > 0).astype(lb.dtype)

        return lb
    
    def _get_selem(self, r):
        # can be used for the dilation, see below
        
        z_to_xy_ratio = self.target_spacing[0] / self.target_spacing[1]
        # radius in different directions
        rz = int(r/z_to_xy_ratio + 0.5)
        rxy = int(r)
        # set up ball
        selem = (np.sum(np.stack(np.meshgrid(*[np.linspace(-1, 1, 2*R+1) for R in [rz, rxy, rxy]], indexing='ij'))**2,0)<=1).astype(float)
        selem /= selem.sum()
        
        selem = torch.from_numpy(selem).cuda().unsqueeze(0).unsqueeze(0).type(torch.float)
        
        return selem
    
    def _dial(self, pred):
        # dialation to increase a segmentation mask from a previous stage
        if not torch.is_tensor(pred):
            pred = torch.from_numpy(pred)
            if torch.cuda.is_available():
                pred = pred.cuda()
        
        if len(pred.shape) < 5:
            while len(pred.shape) < 5:
                pred = pred.unsqueeze(0)
        
        
        z_to_xy_ratio = self.target_spacing[0] / self.target_spacing[1]
        # radius in different directions
        rz = int(self.r_dial_mask/z_to_xy_ratio + 0.5)
        rxy = int(self.r_dial_mask)

        pred_conv = torch.nn.functional.conv3d(pred.type(torch.float),
                                               self._get_selem(self.r_dial_mask),
                                               padding=(rz,rxy,rxy))
    
        pred_dial = (pred_conv > 0).type(torch.float)
        
        return pred_dial
    
    def is_preprocessed_data_tpl(self, data_tpl):
        return 'orig_shape' in data_tpl

    def __call__(self, data_tpl, preprocess_only_im=False, return_np=False):

        spacing = data_tpl['spacing'] if 'spacing' in data_tpl else None
        if 'image' not in data_tpl:
            raise ValueError('No \'image\' found in data_tpl')
        
        # get volume of all information in batch form
        xb = self.get_xb_from_data_tpl(data_tpl, preprocess_only_im)
            
        # now do the preprocessing
        if not torch.cuda.is_available():
            # the preprocessing is also faster in scipy then it is in torch using the CPU
            xb_prep = self.np_preprocessing(xb, spacing)[0]
        else:
            # when CUDA is available we will try to preprocess the data tuple on the GPU...
            xb_cuda = torch.from_numpy(xb).type(torch.float).cuda()
            try:
                xb_prep = self.torch_preprocessing(xb_cuda, spacing)[0]
                if return_np:
                    xb_prep = xb_prep.cpu().numpy()
            except RuntimeError:
                #... unless it fails for a RuntimeError then we will try again on the CPU
                print('Ooops! It seems like your GPU has gone out of memory while trying to '
                      'resize a large volume ({}), trying again on the CPU.'
                      ''.format(list(xb_cuda.shape)))
                torch.cuda.empty_cache()
                xb_prep = self.np_preprocessing(xb, spacing)[0]
                if not return_np:
                    # if we don't want to return the numpy array we're brining it
                    # back to the GPU
                    xb_prep = torch.from_numpy(xb_prep).type(torch.float).cuda()
            
        return xb_prep


    def get_xb_from_data_tpl(self, data_tpl, get_only_im=False):
        
        # getting the image
        xb = data_tpl['image'].astype(float)

        # assuring the array is 4d
        xb = maybe_add_channel_dim(xb)

        if self.has_ps_input:
            # cascade where previous prediction is given as input
            assert self.key_ps_input in data_tpl, 'prediction from previous stage missing'
            pred = data_tpl[self.key_ps_input]
            if torch.is_tensor(pred):
                pred = pred.cpu().numpy()
            # ensure the array is 4d
            pred = maybe_add_channel_dim(pred)
            
            if self.lb_classes is not None:
                pred = reduce_classes(pred, self.lb_classes, self.reduce_lb_to_single_class)
            
            if pred.max() > 1:
                raise NotImplementedError('Didn\'t implement the casacde for multiclass'
                                          'prev stages. Add one hot encoding.')
            
            # the input will be in second position in the array after the image
            xb = np.concatenate([xb, pred])
            
        if self.has_ps_mask:
            # cascade where previous prediction is given as input
            assert self.key_ps_mask in data_tpl, 'prediction from previous stage missing'
            mask = maybe_add_channel_dim(data_tpl[self.key_ps_mask])
            
            if self.r_dial_mask > 0:
                mask = self._dial(mask > 0)[0]
            
            if torch.is_tensor(mask):
                mask = mask.cpu().numpy()
            
            # make sure the mask is binary (if no dialation was applied)
            mask = (mask > 0).astype(xb.dtype)
            
            # the mask will be in the second last position in the array
            # right before the label
            xb = np.concatenate([xb, mask])

        # finally the segmentation
        if 'label' in data_tpl and not get_only_im:     
            # get the label from the data_tpl and clean if applicable
            lb = self.maybe_clean_label_from_data_tpl(data_tpl)

            assert len(lb.shape) == 3, 'label must be 3d'
            lb = lb[np.newaxis].astype(float)
            xb = np.concatenate([xb, lb])
        
        # finally add batch axis
        xb = xb[np.newaxis]

        return xb

    def preprocess_raw_data(self,
                            raw_data,
                            preprocessed_name,
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

        print()
        raw_ds_list = []
        for raw_name in raw_data:
            print('Converting ' + raw_name)
            raw_ds = raw_Dataset(join(environ['OV_DATA_BASE'], 'raw_data', raw_name),
                                 image_folder=image_folder,
                                 dcm_revers=dcm_revers,
                                 dcm_names_dict=dcm_names_dict,
                                 prev_stages=self.prev_stages,
                                 create_missing_labels_as_zero=True)
            raw_ds_list.append(raw_ds)

        if not self.is_initalised:
            print('Preprocessing classes were not initialised when calling '
                  '\'preprocess_raw_data\'. Doing it now.\n')
            self.initialise_preprocessing()

        im_dtype = np.float16 if save_as_fp16 else np.float32

        if data_name is None:
            data_name = '_'.join(sorted(raw_data))

        # root folder of all saved preprocessed data
        outfolder = join(OV_PREPROCESSED, data_name, preprocessed_name)
        plot_folder = join(environ['OV_DATA_BASE'], 'plots', data_name, preprocessed_name)
        print(outfolder, plot_folder)
        # now let's create the output folders
        for f in ['images', 'labels', 'fingerprints']:
            maybe_create_path(join(outfolder, f))
        if self.has_ps_input:
            maybe_create_path(join(outfolder, 'prev_preds'))
        if self.has_ps_mask:
            maybe_create_path(join(outfolder, 'masks'))
        maybe_create_path(plot_folder)

        # Let's quickly store the parameters so we can check later
        # what we've done here.
        self.maybe_save_preprocessing_parameters(outfolder)
        # here is the fun
        for raw_ds in raw_ds_list:
            print()
            sleep(1)
            for i in tqdm(range(len(raw_ds))):
                # read files
                data_tpl = raw_ds[i]

                im, spacing = data_tpl['image'], data_tpl['spacing']

                orig_shape = im.shape[-3:]
                orig_spacing = spacing.copy()

                # some data_tpls come without labels e.g. from dcms if there are no ROIs
                if 'label' not in data_tpl:
                    data_tpl['label'] = np.zeros(orig_shape)

                # get the preprocessed volumes from the data_tpl
                xb = self.__call__(data_tpl, return_np=True)
                im = xb[:self.n_im_channels].astype(im_dtype)
                
                if self.has_ps_input:
                    prev_pred = xb[self.n_im_channels:self.n_im_channels+1]
                
                if self.has_ps_mask:
                    mask = xb[-2:-1]
                
                lb = xb[-1:].astype(np.uint8)

                if lb.max() == 0 and self.save_only_fg_scans:
                    continue

                spacing = self.target_spacing if self.apply_resizing else spacing
                if self.apply_pooling:
                    spacing = np.array(spacing) * np.array(self.pooling_stride)
                # the fingerprints are defined as everything that is left in the data_tpl
                # that is not image, label or prediction from a previous stage
                fingerprint_keys = [key for key in data_tpl if key not in ['image', 'label']]
                fingerprint_keys = [key for key in fingerprint_keys
                                    if not key.startswith('prediction')]
                fingerprint = {key: data_tpl[key] for key in fingerprint_keys}
                fingerprint['orig_shape'] = orig_shape
                fingerprint['orig_spacing'] = orig_spacing
                fingerprint['spacing'] = spacing
                scan = data_tpl['scan']
                if 'dataset' not in fingerprint:
                    fingerprint['dataset'] = raw_name
                if 'pat_id' not in fingerprint:
                    fingerprint['pat_id'] = scan
                # save image and label
                # remeber that the label carries all labels incl. potential masks or
                # predictions from previous stages
                np.save(join(outfolder, 'images', scan), im)
                if self.has_ps_input:
                    np.save(join(outfolder, 'prev_preds', scan), prev_pred)
                if self.has_ps_mask:
                    np.save(join(outfolder, 'masks', scan), mask)

                np.save(join(outfolder, 'labels', scan), lb)
                np.save(join(outfolder, 'fingerprints', scan), fingerprint)

                # additionally do some plots
                im = im.astype(float)
                lb_ch = lb.shape[0]
                im_ch = im.shape[0]

                # get z values of interesting slices
                contains = np.where(np.sum(lb[-1], (1, 2)))[0]
                z_list = [np.argmax(np.sum(lb[-1], (1, 2)))]
                s_list = ['_largest', '_random']
                
                colors = ['red', 'blue', 'green', 'yellow']
                
                if len(contains) > 0:
                    z_list.extend(np.random.choice(contains, size=1))
                else:
                    z_list.extend(np.random.randint(lb.shape, size=1))
                for z, s in zip(z_list, s_list):
                    fig = plt.figure()
                    for ic in range(im_ch):
                        plt.subplot(1, im_ch, ic+1)
                        plt.imshow(im[ic, z], cmap='gray')
                        if lb[0, z].max() > 0:
                            # this if is purely to avoid annoying UserWarning messages that
                            # interrupt the beautiful beautiful tqdm bar
                            plt.contour(lb[0, z] > 0,
                                        linewidths=0.5,
                                        colors=colors[0],
                                        linestyles='dashed')
                        if self.has_ps_input:
                            if prev_pred[0, z].max() > 0:
                                # this if is purely to avoid annoying UserWarning messages that
                                # interrupt the beautiful beautiful tqdm bar
                                plt.contour(prev_pred[0, z] > 0,
                                            linewidths=0.5,
                                            colors=colors[1],
                                            linestyles='dashed')

                        if self.has_ps_mask:
                            if mask[0, z].max() > 0:
                                # this if is purely to avoid annoying UserWarning messages that
                                # interrupt the beautiful beautiful tqdm bar
                                plt.contour(mask[0, z] > 0,
                                            linewidths=0.5,
                                            colors=colors[2],
                                            linestyles='dashed')

                        plt.axis('off')
                    plt.savefig(join(plot_folder, scan + s + '.png'))
                    plt.close(fig)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print('Preprocessing done!')
        
        print('Here are the preprocessing paramters:')
        
        path_to_params = join(outfolder, 'preprocessing_parameters.txt')
        
        with open(path_to_params, 'r') as file:
            print(file.read())

    def plan_preprocessing_raw_data(self, raw_data,
                                    percentiles=[0.5, 99.5],
                                    image_folder=None,
                                    dcm_revers=True,
                                    dcm_names_dict=None,
                                    force_planning=False):

        if isinstance(raw_data, str):
            raw_data = [raw_data]
        elif not isinstance(raw_data, (tuple, list)):
            raise ValueError('raw_data must be str if only infered from a sinlge folder or '
                             'list/tuple.')

        if self.check_parameters() and not force_planning:
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
                                 dcm_names_dict=dcm_names_dict,
                                 create_missing_labels_as_zero=True)
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
        self.dataset_properties['n_fg_classes'] = int(n_fg_classes)

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
                 n_im_channels: int = 1,
                 do_nn_img_interp=False,
                 is_2d=False):
        super().__init__()
        self.apply_resizing = apply_resizing
        self.apply_pooling = apply_pooling
        self.apply_windowing = apply_windowing
        self.n_im_channels = n_im_channels

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
                 n_im_channels: int = 1,
                 do_nn_img_interp=False,
                 is_2d=False):
        super().__init__()
        self.apply_resizing = apply_resizing
        self.apply_pooling = apply_pooling
        self.apply_windowing = apply_windowing
        self.n_im_channels = n_im_channels

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
