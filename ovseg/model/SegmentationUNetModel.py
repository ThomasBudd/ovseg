from ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
from ovseg.augmentation.SegmentationAugmentation import SegmentationAugmentation
from ovseg.data.SegmentationData import SegmentationData
from ovseg.networks.UNet import UNet
from ovseg.training.SegmentationTraining import SegmentationTraining
from ovseg.model.ModelBase import ModelBase
from ovseg.utils.torch_np_utils import check_type
from ovseg.postprocessing.SegmentationPostprocessing import SegmentationPostprocessing
from ovseg.utils.io import save_nii, save_pkl, load_pkl, read_dcms
from ovseg.utils.eval_predictions import eval_prediction_segmentation
from torch.nn import functional as F
import torch
import numpy as np
from os import listdir, mkdir
from os.path import join, exists, basename
import matplotlib.pyplot as plt
from tqdm import tqdm


class SegmentationUNetModel(ModelBase):
    '''
    This model is for 3d medical segmenatation. The networks is chosen to be
    a UNet and patch based input (2d or 3d). The prediction is based on the
    sliding window approach.
    '''

    def __init__(self, val_fold: int, data_name: str, model_name: str,
                 model_parameters=None, preprocessed_name=None,
                 network_name='network', is_inference_only: bool = False,
                 fmt_write='{:.4f}', batch_size_val=1, fp16_val=True):
        super().__init__(self, val_fold=val_fold,
                         data_name=data_name,
                         model_name=model_name,
                         model_parameters=model_parameters,
                         preprocessed_name=preprocessed_name,
                         network_name=network_name,
                         is_inference_only=is_inference_only,
                         fmt_write=fmt_write)
        self.batch_size_val = batch_size_val
        if self.batch_size_val != 1:
            print('Warning: Batch size in validation not implemented yet. Using 1.')
        self.fp16_val = fp16_val

        # get the device
        if not hasattr(self, 'dev'):
            self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'

        if 'prediction' not in self.model_parameters:
            print('no prediction was initialised. use mode \'flip\'')
            self.model_parameters['prediction'] = {'mode': 'flip'}
            if self.parameters_match_saved_ones:
                self.save_model_parameters()
        elif 'mode' not in self.model_parameters['prediction']:
            print('no prediction mode was initialised. use \'flip\'')
            self.model_parameters['prediction']['mode'] = 'flip'
            if self.parameters_match_saved_ones:
                self.save_model_parameters()

    def initialise_preprocessing(self):
        if 'preprocessing' not in self.model_parameters:
            print('No preprocessing parameters found in model_parameters. '
                  'Trying to load from preprocessed_folder...\n')
            if not hasattr(self, 'preprocessed_path'):
                raise AttributeError('preprocessed_path wasn\'t initialiased. '
                                     'Make sure to either pass the '
                                     'preprocessing parameters or the path '
                                     'to the preprocessed folder were an '
                                     'extra copy is stored.')
            else:
                print('Loaded preprocessing parameters and updating model '
                      'parameters. Consider saving them again to keep '
                      'preprocessing parameters for the future!\n')
                prep_params = load_pkl(join(self.preprocessed_path,
                                            'preprocessing_parameters.pkl'))
                self.model_parameters.update({'preprocessing': prep_params})

                if self.parameters_match_saved_ones:
                    # when there was no conflict at loading time we can savely
                    # resave the updated model parameters.
                    self.save_model_parameters()

        params = self.model_parameters['preprocessing'].copy()
        self.preprocessing = SegmentationPreprocessing(**params)

        # if the preprocessing method was not initialised with the full set
        # of parameters and the other model paramters match the ones found in
        # the saved file we can savely add the loaded preprocessing parameters
        if self.preprocessing.attributes_loaded and self.parameters_match_saved_ones:

            # add full preprocessing parameters to the model parameters
            self.model_parameters['preprocessing'] = self.preprocessing.get_attributes()
            self.save_model_parameters()
        print('Preprocessing initialised.\n')

    def initialise_augmentation(self):

        # first initialise CPU augmentation
        # this happens in the dataloader
        if self.preprocessing.target_spacing is not None and self.preprocessing.apply_resizing:
            spacing = self.preprocessing.target_spacing
        if 'cpu_augmentation' not in self.model_parameters:
            print('no \'cpu_augmentation\' parameters found. '
                  'performing no augmentation on the CPU.\n')
        else:
            print('CPU augmentation parameters found!\n')
            params = self.model_parameters['cpu_augmentation'].copy()
            for key in params:
                if not hasattr(params[key], 'spacing'):
                    params[key]['spacing'] = spacing
                if self.parameters_match_saved_ones:
                    self.save_model_parameters()
            self.cpu_augmentation = SegmentationAugmentation(params)

        # now GPU augmentation. We will hand this to the trainier and
        # do it on the fly before executing each loaded batch
        if 'gpu_augmentation' not in self.model_parameters:
            print('no \'gpu_augmentation\' parameters found. '
                  'performing no augmentation on the GPU\n.')
            self.gpu_augmentation = None
        else:
            print('GPU augmentation parameters found!\n')
            for key in params:
                if not hasattr(params[key], 'spacing'):
                    params[key]['spacing'] = spacing
            params = self.model_parameters['gpu_augmentation'].copy()
            self.gpu_augmentation = SegmentationAugmentation(params)

        print('Augmentation initialised')

    def initialise_network(self):
        if 'network' not in self.model_parameters:
            raise AttributeError('model_parameters must have key '
                                 '\'network\'. These must contain the '
                                 'dict of network paramters.')
        params = self.model_parameters['network'].copy()
        self.network = UNet(**params).to(self.dev)
        print('Network initialised.\n')

    def initialise_postprocessing(self):
        try:
            params = self.model_parameters['postprocessing'].copy()
        except KeyError:
            print('No parameter for postprocessing were given. Take defaut '
                  'values (no removing of small connected components.\n')
            params = {}
        self.postprocessing = SegmentationPostprocessing(**params)
        print('Postprocessing initialised.\n')

    def initialise_data(self):
        if 'data' not in self.model_parameters:
            raise AttributeError('model_parameters must have key '
                                 '\'data\'. These must contain the '
                                 'dict of training paramters.')

        # Let's get the parameters and add the cpu augmentation
        params = self.model_parameters['data'].copy()
        if hasattr(self, 'cpu_augmentation'):
            try:
                params['trn_dl_params']['augmentation'] = \
                    self.cpu_augmentation
            except KeyError:
                raise KeyError('No \'trn_dl_params\' found in \'data\' key of '
                               'model_parameters. This information is '
                               'essential to create the data object.')

            try:
                params['val_dl_params']['augmentation'] = \
                    self.cpu_augmentation
            except KeyError:
                print('No \'val_dl_params\' found in \'data\' key of '
                      'model_parameters. No validation data will be used '
                      'during training.\n')

        self.data = SegmentationData(val_fold=self.val_fold,
                                     preprocessed_path=self.preprocessed_path,
                                     **params)
        print('Data initialised.\n')

    def initialise_training(self):
        if 'training' not in self.model_parameters:
            raise AttributeError('model_parameters must have key '
                                 '\'training\'. These must contain the '
                                 'dict of training paramters.\n')

        # create the training module by concatenating the gup augmentation and the network
        self.training_module = torch.nn.Sequential(self.augmentation.gpu_augmentation.augment_batch,
                                                   self.network)
        params = self.model_parameters['training'].copy()
        self.training = SegmentationTraining(module=self.training_module,
                                             trn_dl=self.data.trn_dl,
                                             val_dl=self.data.val_dl,
                                             model_path=self.model_path,
                                             network_name=self.network_name,
                                             **params)
        print('Training initialised.\n')

    def _sliding_window(self, volume, overlap=0.5, window=1):
        '''
        does one complete slide over the whole volume to average the
        predictions
        '''

        if not torch.is_tensor(volume):
            raise TypeError('Input must be torch tensor')
        if not len(volume.shape) == 4:
            raise ValueError('Volume must be a 4d tensor (incl channel axis)')

        # in case the volume is smaller than the patch size we pad it
        # and save the input size to crop again before returning
        shape_in = volume.shape

        # %% check the window and patch size
        # the window must be a positive number or an array of only positives
        if np.isscalar(window) or isinstance(window, np.ndarray):
            if np.max(window) <= 0:
                raise ValueError('Window must be positive')
        elif torch.is_tensor(window):
            if window.max().item() <= 0:
                raise ValueError('Window must be positive')
        else:
            raise TypeError('window must be np.ndarray torch.tensor or scalar')

        # let's get the patch size we use for evaluation
        patch_size = self.model_parameters['data']['trn_dl_params']['patch_size']
        patch_size = np.array(patch_size).astype(int)

        # if window is not scalar we need the shape to match the patch size
        if isinstance(window, np.ndarray) or torch.is_tensor(window):
            assert np.all(window.shape == patch_size)

            # we will do everything in torch from here
            window = torch.tensor(window).to(self.dev)

            # adding a channel axis as this makes things easier later
            window = window.unsqueeze(0)

        # for convinience simply
        if len(patch_size) == 2:
            is_2d = True
            patch_size = np.concatenate([patch_size, [1]])
            if not np.isscalar(window):
                window = window.unsqueeze(-1)
        else:
            is_2d = False

        # %% possible padding of too small volumes
        pad = [0, patch_size[2] - shape_in[3], 0, patch_size[1] - shape_in[2],
               0, patch_size[0] - shape_in[1]]
        pad = np.maximum(pad, 0).tolist()
        volume = F.pad(volume, pad)
        nx, ny, nz = volume.shape[1:]
        n_ch = self.model_parameters['network']['out_channels']

        # %% reserve storage
        pred = torch.zeros((n_ch, nx, ny, nz), device=self.dev)
        ovlp = torch.zeros_like(pred)

        # upper left corners of all patches
        x_list = list(range(0, nx - patch_size[0],
                            int(patch_size[0] * overlap))) + [nx - patch_size[0]]
        y_list = list(range(0, ny - patch_size[1],
                            int(patch_size[1] * overlap))) + [ny - patch_size[1]]
        z_list = list(range(0, nz - patch_size[2],
                            max([int(patch_size[2] * overlap), 1]))) + [nz - patch_size[2]]
        xyz_list = []
        for x in x_list:
            for y in y_list:
                for z in z_list:
                    xyz_list.append((x, y, z))

        # %% now the magic!
        with torch.no_grad():
            for x, y, z in xyz_list:
                # crop
                batch = volume[:, x:x+patch_size[0], y:y+patch_size[1],
                               z:z+patch_size[2]]
                # add batch axis
                batch = batch.unsqueeze(0)

                # remove z axis if we have 2d prediction
                batch = batch[..., 0] if is_2d else batch
                out = self.network(batch)[0][0]

                # add z axis again maybe
                out = out.unsqueeze(-1) if is_2d else out

                # update pred and overlap
                pred[:, x:x+patch_size[0], y:y+patch_size[1],
                     z:z+patch_size[2]] += out * window
                ovlp[:, x:x+patch_size[0], y:y+patch_size[1],
                     z:z+patch_size[2]] += window

        # %% bring maybe back to old shape and possible to numpy
        pred = pred[:, :shape_in[1], :shape_in[2], :shape_in[3]]
        ovlp = ovlp[:, :shape_in[1], :shape_in[2], :shape_in[3]]

        # let's get out of here
        return pred, ovlp

    def _eval_network_flip(self, volume):

        if not torch.is_tensor(volume):
            raise TypeError('Input must be torch tensor')
        if not len(volume.shape) == 4:
            raise ValueError('Volume must be a 4d tensor (incl channel axis)')
        # now we make the list of all flippings
        is_2d = len(self.model_parameters['data']['trn_dl_params']
                    ['patch_size']) == 2
        flip_list = []
        flip_z_list = [False] if is_2d else [True, False]
        for fx in [True, False]:
            for fy in [True, False]:
                for fz in flip_z_list:
                    flip = []
                    if fx:
                        flip.append(1)
                    if fy:
                        flip.append(2)
                    if fz:
                        flip.append(3)
                    flip_list.append(flip)

        # let's iterate over these
        preds, ovlps = [], []
        for flip in flip_list:

            if len(flip) > 0:
                volume = volume.flip(flip)

            # predict flipped volume
            pred, ovlp = self._sliding_window(volume)

            # and flip back
            if len(flip) > 0:
                pred = pred.flip(flip)
                ovlp = ovlp.flip(flip)
                volume = volume.flip(flip)
            preds.append(pred)
            ovlps.append(ovlp)

        # now take the mean over these tensors
        pred = torch.stack(preds).mean(0)
        ovlp = torch.stack(ovlps).mean(0)

        return pred, ovlp

    def _eval_network_tta(self, im):
        raise NotImplementedError('TTA not implemented yet.')

    def predict(self, data, do_preprocessing=False, do_postprocessing=True,
                to_original_shape=False):
        '''
        There are a lot of differnt ways to do prediction. Some do require direct preprocessing
        some don't need the postprocessing imidiately (e.g. when ensembling)
        Same holds for the resizing to original shape. In the validation case we wan't to apply
        some postprocessing (argmax and removing of small lesions) but not the resizing.
        '''
        self.network = self.network.eval()
        im = data['image']
        is_np,  _ = check_type(im)
        if is_np:
            im = torch.from_numpy(im).to(self.dev)
        else:
            im = im.to(self.dev)

        # most of the time we expect this to be False
        # Validation scans should be prepocessed and when we perform ensembling to testing
        # we would only preprocess the data once and then hand these images in each model
        if do_preprocessing:
            im = self.preprocessing(im, data['spacing'])

        # no the importat part: the sliding window evaluation (or derivatices of it)
        if self.fp16_val and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                pred = self._eval_network_on_full_volume(im)
        else:
            pred = self._eval_network_on_full_volume(im)

        if do_postprocessing:
            orig_shape = data['orgi_shape'] if to_original_shape else None
            pred = self.postprocessing(pred, orig_shape)

        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()

        return pred

    def predict_from_dcm(self, dcm_folder):
        data = read_dcms(dcm_folder)
        return self.predict(data, True, True, True)

    def _eval_network_on_full_volume(self, volume):

        # first let's process the volume and check that it is right
        is_np, _ = check_type(volume)
        if is_np:
            # we will operate only in torch in here
            volume = torch.from_numpy(volume)
        volume = volume.to(self.dev)

        # we need 4d input here, or 3d and add the channel axes
        if len(volume.shape) == 3:
            volume = volume.unsqueeze(0)
        elif not len(volume.shape) == 4:
            raise ValueError('Input volume must be 3d or 4d.')

        # check which mode we use for prediction
        assert self.model_parameters['prediction']['mode'].lower() in ['simple', 'flip', 'tta']

        # here is the fun part!
        if self.model_parameters['prediction']['mode'].lower() == 'simple':
            pred, ovlp = self._sliding_window(volume)
        elif self.model_parameters['prediction']['mode'].lower() == 'flip':
            pred, ovlp = self._eval_network_flip(volume)
        elif self.model_parameters['prediction']['mode'].lower() == 'tta':
            pred, ovlp = self._eval_network_tta(volume)

        assert ovlp.max() > 0
        # back to probabilities
        pred = pred/ovlp
        # if the input was numpy the output will be as well.
        if is_np:
            pred = pred.cpu().numpy()
        return pred

    def save_prediction(self, pred, data, pred_folder, name):

        out_file = join(pred_folder, name+'.nii.gz')

        if self.preprocessing.apply_resizing:
            spacing = self.preprocessing.target_spacing
        else:
            spacing = data['spacing']

        save_nii(pred, out_file, spacing)

    def plot_prediction(self, pred, data, plot_folder, name):
        '''
        We're not diferentiatig between different fg classes for now.
        '''
        im = data['image']
        seg = (data['label'] > 0).astype(float)
        pred = (pred.copy() > 0).astype(float)

        contains = np.where(np.sum(seg, (0, 1)))[0]
        if len(contains) == 0:
            return
        z_max = np.argmax(np.sum(seg, (0, 1)))
        z_random = np.random.choice(contains)

        if not hasattr(self, 'plot_ovlp'):
            print('Variable \'plot_ovlp\' was not initialised for this model.'
                  'Setting it to 1.')
            self.plot_ovlp = 1

        # create the ovlp arrays
        seg_ovlp = np.stack([im + self.plot_ovlp * seg, im, im], -1)
        seg_ovlp = (seg_ovlp - seg_ovlp.min()) / (seg_ovlp.max() - seg_ovlp.min())
        pred_ovlp = np.stack([im + self.plot_ovlp * pred, im, im], -1)
        pred_ovlp = (pred_ovlp - pred_ovlp.min()) / (pred_ovlp.max() - pred_ovlp.min())

        # now plot largest and random slice
        fig = plt.figure()
        plt.subplot(121)
        plt.title('Ground truth')
        plt.imshow(seg_ovlp[:, :, z_max])
        plt.axis('off')
        plt.subplot(122)
        plt.title('prediction')
        plt.imshow(pred_ovlp[:, :, z_max])
        plt.axis('off')
        plt.savefig(join(plot_folder, name+'largest.png'))
        plt.close(fig)
        fig = plt.figure()
        plt.subplot(121)
        plt.title('Ground truth')
        plt.imshow(seg_ovlp[:, :, z_random])
        plt.axis('off')
        plt.subplot(122)
        plt.title('prediction')
        plt.imshow(pred_ovlp[:, :, z_random])
        plt.axis('off')
        plt.savefig(join(plot_folder, name+'random.png'))
        plt.close(fig)

    def compute_error_metrics(self, pred, data):
        return eval_prediction_segmentation(data['label'], pred)
