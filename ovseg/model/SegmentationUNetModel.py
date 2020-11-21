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
from ovseg.utils.grid_utils import get_centred_np_grid
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
        super().__init__(val_fold=val_fold,
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

        # now prepare some stuff we need for the prediction
        self.n_ch = self.model_parameters['network']['out_channels']
        self.is_2d = len(self.model_parameters['data']['trn_dl_params']['patch_size']) == 2
        self.patch_size = np.array(self.model_parameters['data']['trn_dl_params']['patch_size'])
        self.patch_size = self.patch_size.astype(int)
        if self.is_2d:
            self.patch_size = np.concatenate([self.patch_size, [1]])

        if 'prediction' not in self.model_parameters:
            print('no prediction was initialised. use mode \'flip\'')
            self.model_parameters['prediction'] = {'mode': 'flip'}
            if self.parameters_match_saved_ones:
                self.save_model_parameters()
        elif 'mode' not in self.model_parameters['prediction']:
            print('no prediction mode was initialised. use \'simple\'')
            self.model_parameters['prediction']['mode'] = 'simple'
            if self.parameters_match_saved_ones:
                self.save_model_parameters()

        # at the edges of the patch we know that the prediction quality is worse then in the center
        # so it makes sense to use some gaussian weighting against this.
        # however this also means that one has to determine a sigme for this
        if 'patch_weight_type' not in self.model_parameters['prediction']:
            print('No patch_weight_type specified in the model parameters. Using \'constant\'.')
            self.model_parameters['prediction']['patch_weight_type'] = 'constant'
            if self.parameters_match_saved_ones:
                self.save_model_parameters()

        pwt = self.model_parameters['prediction']['patch_weight_type']
        if pwt.lower() == 'constant':
            self.prediction_patch_weight = np.ones(self.patch_size)
        elif pwt.lower() == 'gaussian':
            # for the gaussian patch weights we build up the weight function here
            sigma = self.model_parameters['prediction']['sigma_gaussian']
            # length of each axis in real world coordinates
            grid = get_centred_np_grid(self.patch_size, self.spacing)
            norm_x = np.sum(grid**2, 0)

            # the magic formula
            self.prediction_patch_weight = np.exp(-0.5 * norm_x/sigma**2)

            if self.prediction_patch_weight.min() == 0:
                raise ValueError('0 occured in the patch weight when computing the Gaussian ovlp. '
                                 'Choose larger sigma')
        else:
            print('Value Error: '+pwt+' was not recognised as a patch_weight_type. Please check '
                  'the implementation or add your own patch_weight_type. This model will crash '
                  'in prediction')
        # adding the channel axes
        self.prediction_patch_weight = self.prediction_patch_weight[np.newaxis]
        # as all the prediction happens in torch we can put the weight there
        self.prediction_patch_weight = torch.from_numpy(self.prediction_patch_weight).to(self.dev)

        # now the overlap with which we slide the windows
        if 'overlap' not in self.model_parameters['prediction']:
            print('No overlap parameter was specified. Chooing 0.5')
            self.prediction_overlap = 0.5
            if self.parameters_match_saved_ones:
                self.save_model_parameters()
        else:
            self.prediction_overlap = self.model_parameters['prediction']['overlap']

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

        params = self.model_parameters['preprocessing']
        print(self.preprocessed_name)
        self.preprocessing = SegmentationPreprocessing(data_name=self.data_name,
                                                       preprocessed_name=self.preprocessed_name,
                                                       **params)

        # if the preprocessing method was not initialised with the full set
        # of parameters and the other model paramters match the ones found in
        # the saved file we can savely add the loaded preprocessing parameters
        if self.preprocessing.attributes_loaded and self.parameters_match_saved_ones:

            # add full preprocessing parameters to the model parameters
            self.model_parameters['preprocessing'] = self.preprocessing.get_attributes()
            self.save_model_parameters()

        # now let's set the spacing we get from the preprocessing
        if self.preprocessing.target_spacing is not None and self.preprocessing.apply_resizing:
            self.spacing = np.array(self.preprocessing.target_spacing)
        else:
            print('Spacing wasn\'t found at the preprocessing module. Using isotropic spacing '
                  '(1, 1, 1).')
            self.spacing = np.array([1, 1, 1])
        print('Preprocessing initialised.\n')

    def initialise_augmentation(self):

        # first initialise CPU augmentation
        # this happens in the dataloader
        if 'cpu_augmentation' not in self.model_parameters:
            print('no \'cpu_augmentation\' parameters found. '
                  'performing no augmentation on the CPU.\n')
            params = {}
        else:
            print('CPU augmentation parameters found!\n')
            params = self.model_parameters['cpu_augmentation']
            for key in params:
                if not hasattr(params[key], 'spacing'):
                    params[key]['spacing'] = self.spacing
                if self.parameters_match_saved_ones:
                    self.save_model_parameters()
        self.cpu_augmentation = SegmentationAugmentation(params)

        # now GPU augmentation. We will hand this to the trainier and
        # do it on the fly before executing each loaded batch
        if 'gpu_augmentation' not in self.model_parameters:
            print('no \'gpu_augmentation\' parameters found. '
                  'performing no augmentation on the GPU\n.')
            params = {}
        else:
            print('GPU augmentation parameters found!\n')
            params = self.model_parameters['gpu_augmentation']
            for key in params:
                if not hasattr(params[key], 'spacing'):
                    params[key]['spacing'] = self.spacing
        self.gpu_augmentation = SegmentationAugmentation(params)

        # just in case we added some spacings here
        if self.parameters_match_saved_ones:
            self.save_model_parameters()

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

    def _sliding_window(self, volume, ROI=None):
        '''
        uses sliding window over the volume. Overlap of the winodws and patch_weight
        must be specified in the model_parameters.
        INPUT:
            volume - 3d or 4d np array or tensor
            ROI - 3d boolean array
                  predict only patches that have at least one True voxel
                  default: all True
        OUTPUT:
            pred - summed up overlapping weighted patch values
            ovlp - keeps record of how much overlap we had in each voxel
        '''

        if not torch.is_tensor(volume):
            raise TypeError('Input must be torch tensor')
        if not len(volume.shape) == 4:
            raise ValueError('Volume must be a 4d tensor (incl channel axis)')

        # in case the volume is smaller than the patch size we pad it
        # and save the input size to crop again before returning
        shape_in = volume.shape

        # %% possible padding of too small volumes
        pad = [0, self.patch_size[2] - shape_in[3], 0, self.patch_size[1] - shape_in[2],
               0, self.patch_size[0] - shape_in[1]]
        pad = np.maximum(pad, 0).tolist()
        volume = F.pad(volume, pad)
        nx, ny, nz = volume.shape[1:]

        # %% reserve storage
        pred = torch.zeros((self.n_ch, nx, ny, nz), device=self.dev)
        ovlp = torch.zeros((1, nx, ny, nz), device=self.dev)
        if ROI is None:
            # if the ROI
            ROI = torch.ones((nx, ny, nz)) > 0

        # upper left corners of all patches
        x_list = list(range(0, nx - self.patch_size[0],
                            int(self.patch_size[0] * self.prediction_overlap))) \
            + [nx - self.patch_size[0]]
        y_list = list(range(0, ny - self.patch_size[1],
                            int(self.patch_size[1] * self.prediction_overlap))) \
            + [ny - self.patch_size[1]]
        z_list = list(range(0, nz - self.patch_size[2],
                            max([int(self.patch_size[2] * self.prediction_overlap), 1]))) \
            + [nz - self.patch_size[2]]
        xyz_list = []
        for x in x_list:
            for y in y_list:
                for z in z_list:
                    # we only predict the patch if we intersect the ROI
                    if ROI[x:x+self.patch_size[0], y:y+self.patch_size[1],
                           z:z+self.patch_size[2]].any().item():
                        xyz_list.append((x, y, z))

        # introduce batch size
        n_full_batches = len(xyz_list) // self.batch_size_val
        xyz_batched = [xyz_list[i * self.batch_size_val: (i + 1) * self.batch_size_val]
                       for i in range(n_full_batches)]
        if n_full_batches * self.batch_size_val < len(xyz_list):
            xyz_batched.append(xyz_list[n_full_batches * self.batch_size_val:])
        print(len(xyz_batched))
        # %% now the magic!
        with torch.no_grad():
            for xyz_batch in xyz_batched:
                # crop
                batch = torch.stack([volume[:, x:x+self.patch_size[0], y:y+self.patch_size[1],
                                            z:z+self.patch_size[2]] for x, y, z in xyz_batch])

                # remove z axis if we have 2d prediction
                batch = batch[..., 0] if self.is_2d else batch
                # remember that the network is outputting a list of predictions for each scale
                if self.fp16_val and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        out = self.network(batch)[0]
                else:
                    out = self.network(batch)[0]

                # add z axis again maybe
                out = out.unsqueeze(-1) if self.is_2d else out

                # update pred and overlap
                for i, (x, y, z) in enumerate(xyz_batch):
                    pred[:, x:x+self.patch_size[0], y:y+self.patch_size[1],
                         z:z+self.patch_size[2]] += out[i] * self.prediction_patch_weight
                    ovlp[:, x:x+self.patch_size[0], y:y+self.patch_size[1],
                         z:z+self.patch_size[2]] += self.prediction_patch_weight

        # %% bring maybe back to old shape and possible to numpy
        pred = pred[:, :shape_in[1], :shape_in[2], :shape_in[3]]
        ovlp = ovlp[:, :shape_in[1], :shape_in[2], :shape_in[3]]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # note that the network returns logits
        return F.softmax(pred, 0), ovlp

    def _eval_network_flip(self, volume):

        if not torch.is_tensor(volume):
            raise TypeError('Input must be torch tensor')
        if not len(volume.shape) == 4:
            raise ValueError('Volume must be a 4d tensor (incl channel axis)')
        # now we make the list of all flippings
        flip_list = []
        flip_z_list = [False] if self.is_2d else [True, False]
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

    def _eval_network_tta(self, volume):
        if 'eps_tta' not in self.model_parameters['prediction']:
            raise AttributeError('\'eps_tta\' must be initialised as a prediction parameter when '
                                 'using test time augmentations (tta). \'eps_tta\' defines for '
                                 'which difference in prediction a voxel is predicted again.')
        eps = self.model_parameters['prediction']['eps_tta']
        if 'max_it_tta' not in self.model_parameters['prediction']:
            raise AttributeError('\'max_it_tta\' must be initialised as a prediction parameter when'
                                 ' using test time augmentations (tta). \'max_it_tta\' defines how '
                                 'many augmentations we do at most per volume.')
        max_it = self.model_parameters['prediction']['max_it_tta']

        # first prediction over the whole volume
        pred, ovlp = self._sliding_window(volume, ROI=None)
        ROI = (pred / ovlp) > eps

        # now the fancy iteration
        # -1 because we already did one prediction
        for _ in range(max_it - 1):

            # check if we still have voxel where we are unsure
            if not ROI.any().item():
                break

            # else let's go on an augment the volume
            volume_aug = volume.detach().clone()
            volume_aug = self.cpu_augmentation.augment_volume(volume_aug, is_inverse=False)
            volume_aug = self.gpu_augmentation.augment_volume(volume_aug, is_inverse=False)
            print(volume_aug.shape)
            pred_update, ovlp_update = self._sliding_window(volume, ROI=ROI)
            # bring back to original coordinate system
            sample = torch.cat([pred_update, ovlp_update])
            sample = self.gpu_augmentation.augment_volume(sample, is_inverse=True)
            sample = self.cpu_augmentation.augment_volume(sample, is_inverse=True)
            pred_update, ovlp_update = sample[:-1], sample[-1:]

            # first update the ROI
            ROI = (pred/ovlp - (pred + pred_update)/(ovlp + ovlp_update)).abs()
            print(ROI.max())
            ROI = ROI > eps
            # then the prediction and overlap
            pred = pred + pred_update
            ovlp = ovlp + ovlp_update

        return pred, ovlp

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
            im = torch.from_numpy(im.astype(np.float32)).to(self.dev)
        else:
            im = im.to(self.dev)

        # most of the time we expect this to be False
        # Validation scans should be prepocessed and when we perform ensembling to testing
        # we would only preprocess the data once and then hand these images in each model
        if do_preprocessing:
            im = self.preprocessing(im, data['spacing'])

        # no the importat part: the sliding window evaluation (or derivatives of it)
        pred = self._eval_network_on_full_volume(im)

        if do_postprocessing:
            orig_shape = data['orig_shape'] if to_original_shape else None
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
