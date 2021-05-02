import warnings
warnings.simplefilter("ignore", UserWarning)
from ovseg.preprocessing.SegmentationPreprocessing import \
    SegmentationPreprocessing
from ovseg.augmentation.SegmentationAugmentation import \
    SegmentationAugmentation
from ovseg.data.SegmentationData import SegmentationData
from ovseg.prediction.SlidingWindowPrediction import SlidingWindowPrediction
from ovseg.networks.UNet import UNet
from ovseg.networks.iUNet import iUNet
from ovseg.training.SegmentationTraining import SegmentationTraining
from ovseg.model.ModelBase import ModelBase
from ovseg.utils.torch_np_utils import check_type
from ovseg.postprocessing.SegmentationPostprocessing import \
    SegmentationPostprocessing
from ovseg.utils.io import save_nii_from_data_tpl, load_pkl, read_nii
import torch
import numpy as np
from os import environ, makedirs, listdir, mkdir
from os.path import join, basename, exists
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep


class SegmentationModel(ModelBase):
    '''
    This model is for 3d medical segmenatation. The networks is chosen to be
    a UNet and patch based input (2d or 3d). The prediction is based on the
    sliding window approach.
    '''

    def __init__(self, val_fold: int, data_name: str, model_name: str,
                 model_parameters=None, preprocessed_name=None,
                 network_name='network', is_inference_only: bool = False,
                 fmt_write='{:.4f}', model_parameters_name='model_parameters',
                 plot_n_random_slices=1, dont_store_data_in_ram=False):
        self.dont_store_data_in_ram = dont_store_data_in_ram
        super().__init__(val_fold=val_fold, data_name=data_name, model_name=model_name,
                         model_parameters=model_parameters, preprocessed_name=preprocessed_name,
                         network_name=network_name, is_inference_only=is_inference_only,
                         fmt_write=fmt_write, model_parameters_name=model_parameters_name)
        self.initialise_prediction()
        self.plot_n_random_slices = plot_n_random_slices

    def initialise_preprocessing(self):
        if 'preprocessing' not in self.model_parameters:
            print('No preprocessing parameters found in model_parameters. '
                  'Trying to load from preprocessed_folder...')
            if not hasattr(self, 'preprocessed_path'):
                raise AttributeError('preprocessed_path wasn\'t initialiased. '
                                     'Make sure to either pass the '
                                     'preprocessing parameters or the path '
                                     'to the preprocessed folder were an '
                                     'extra copy is stored.')
            else:
                prep_params = load_pkl(join(self.preprocessed_path,
                                            'preprocessing_parameters.pkl'))
                self.model_parameters['preprocessing'] = prep_params
                if self.parameters_match_saved_ones:
                    print('Loaded preprocessing parameters and updating model '
                          'parameters.')
                    self.save_model_parameters()
                else:
                    print('Loaded preprocessing parameters without saving them to the model '
                          'parameters as current model parameters don\'t match saved ones.')
        params = self.model_parameters['preprocessing'].copy()
        self.preprocessing = SegmentationPreprocessing(**params)
        if self.preprocessing.lb_classes is not None:
            self.n_fg_classes = len(self.preprocessing.lb_classes)
        elif self.model_parameters['network']['out_channels'] is not None:
            self.n_fg_classes = self.model_parameters['network']['out_channels'] - 1
        elif hasattr(self.preprocessing, 'dataset_properties'):
            print('Using all foreground classes for computing the DSCS')
            self.n_fg_classes = self.preprocessing.dataset_properties['n_fg_classes']
        else:
            raise AttributeError('Something seems to be wrong. Could not figure out the number '
                                 'of foreground classes in the problem...')

        # now we check if we perform a cascasde:
        if self.is_cascade():
            # in this case we will create a second preprocessing module for the segmentations
            # of the previous stage
            params_ps = {'apply_windowing': False,
                         'scaling': [1, 0],
                         'apply_resizing': params['apply_resizing'],
                         'apply_pooling': params['apply_pooling'],
                         'do_nn_img_interp': True}
            if params_ps['apply_resizing']:
                params_ps['target_spacing'] = params['target_spacing']
            if params_ps['apply_pooling']:
                params_ps['pooling_stride'] = params['pooling_stride']
            self.preprocessing_for_pred_from_prev_stage = SegmentationPreprocessing(**params_ps)

    def initialise_augmentation(self):

        # first initialise CPU augmentation
        # this happens in the dataloader
        if 'augmentation' in self.model_parameters:
            self.augmentation = SegmentationAugmentation(**self.model_parameters['augmentation'])

    def initialise_network(self):
        if 'network' not in self.model_parameters:
            raise AttributeError('model_parameters must have key '
                                 '\'network\'. These must contain the '
                                 'dict of network paramters.')

        if self.model_parameters['network']['out_channels'] is None:
            # we check if no out_channels argument was set. In this case
            # the default is chosen: Number of foreground classes +1
            self.model_parameters['network']['out_channels'] = self.n_fg_classes + 1
            if self.parameters_match_saved_ones:
                self.save_model_parameters()
        params = self.model_parameters['network'].copy()
        if self.model_parameters['architecture'].lower() in ['unet', 'u-net']:
            self.network = UNet(**params).to(self.dev)
        elif self.model_parameters['architecture'].lower() in ['iunet', 'iu-net']:
            self.network = iUNet(**params).to(self.dev)

    def initialise_prediction(self):
        # by default we take the same batch size as we used during training for inference
        # but in theory we should also be able to use a larger one to seep up everything
        params = {'network': self.network,
                  'patch_size': self.model_parameters['data']['trn_dl_params']['patch_size']}
        if 'prediction' not in self.model_parameters:
            print('model_parameters doesn\'t have key \'prediction\' to speficfy how full volumes '
                  'are processed by the model. Using default parameters')
        else:
            params.update(self.model_parameters['prediction'])

        self.prediction = SlidingWindowPrediction(**params)

    def initialise_postprocessing(self):
        try:
            params = self.model_parameters['postprocessing'].copy()
        except KeyError:
            print('No parameter for postprocessing were given. Take default: argmax without '
                  'removing of small connected components.')
            params = {}
        self.postprocessing = SegmentationPostprocessing(**params)

    def initialise_data(self):
        if 'data' not in self.model_parameters:
            raise AttributeError('model_parameters must have key '
                                 '\'data\'. These must contain the '
                                 'dict of training paramters.')

        # Let's get the parameters and add the cpu augmentation
        params = self.model_parameters['data'].copy()
        data_aug = self.augmentation.np_augmentation

        # add augmentation
        for key in ['trn_dl_params', 'val_dl_params']:
            try:
                params[key]['augmentation'] = data_aug
            except KeyError:
                continue

        # if we don't want to store our data in ram...
        if self.dont_store_data_in_ram:
            for key in ['trn_dl_params', 'val_dl_params']:
                params[key]['store_data_in_ram'] = False
                params[key]['store_coords_in_ram'] = False
        self.data = SegmentationData(val_fold=self.val_fold,
                                     preprocessed_path=self.preprocessed_path,
                                     **params)
        print('Data initialised')

    def initialise_training(self):
        if 'training' not in self.model_parameters:
            raise AttributeError('model_parameters must have key '
                                 '\'training\'. These must contain the '
                                 'dict of training paramters.')
        params = self.model_parameters['training'].copy()
        self.training = SegmentationTraining(network=self.network,
                                             trn_dl=self.data.trn_dl,
                                             val_dl=self.data.val_dl,
                                             model_path=self.model_path,
                                             network_name=self.network_name,
                                             augmentation=self.augmentation.torch_augmentation,
                                             **params)

    def is_cascade(self):
        return 'previous_stage' in self.model_parameters

    def predict(self, data_tpl, image_key='image', pred_fps_key='pred_fps'):
        '''
        There are a lot of differnt ways to do prediction. Some do require direct preprocessing
        some don't need the postprocessing imidiately (e.g. when ensembling)
        Same holds for the resizing to original shape. In the validation case we wan't to apply
        some postprocessing (argmax and removing of small lesions) but not the resizing.
        '''
        self.network = self.network.eval()
        im = data_tpl[image_key]
        is_np,  _ = check_type(im)
        is_cascade = pred_fps_key in data_tpl
        if len(im.shape) == 3:
            im = im[np.newaxis] if is_np else im.unsqueeze(0)
        if is_np:
            if is_cascade:
                pred_fps = data_tpl[pred_fps_key]
                if len(pred_fps.shape) == 3:
                    if type(pred_fps) == np.ndarray:
                        pred_fps = pred_fps[np.newaxis]
                        im = np.concatenate([im, pred_fps])
                        im = torch.from_numpy(im).to(self.dev)
                    else:
                        pred_fps = pred_fps.unsqueeze(0).to(self.dev)
                        im = im.to(self.dev)
                        im = torch.cat([im, pred_fps])

        else:
            im = im.to(self.dev)
            if is_cascade:
                pred_fps = data_tpl[pred_fps_key]
                if type(pred_fps) == np.ndarray:
                    pred_fps = torch.from_numpy(pred_fps).to(self.dev)
                if len(pred_fps.shape) == 3:
                    pred_fps = pred_fps.unsqueeze(0)
                im = torch.cat([im, pred_fps])

        # the preprocessing will only do something if the image is not preprocessed yet
        if not self.preprocessing.is_preprocessed_data_tpl(data_tpl):
            im = self.preprocessing(data_tpl, preprocess_only_im=True)

        # let's quickly get back to the prediction from the previous stage
        if is_cascade and self.n_fg_classes > 1:
            im, pred_fps = im[:1], im[1]
            pred_fps = torch.stack([pred_fps == c for c in range(1, self.n_fg_classes + 1)])
            im = torch.cat([im, pred_fps.type(im.dtype)])

        # now the importat part: the sliding window evaluation (or derivatives of it)
        pred = self.prediction(im)
        data_tpl[self.pred_key] = pred

        # inside the postprocessing the result will be attached to the data_tpl
        self.postprocessing.postprocess_data_tpl(data_tpl, self.pred_key)

        return data_tpl[self.pred_key]

    def __call__(self, data_tpl, image_key='image'):
        return self.predict(data_tpl, image_key=image_key)

    def save_prediction(self, data_tpl, folder_name, filename=None):

        # find name of the file
        if filename is None:
            if 'raw_label_file' in data_tpl:
                filename = basename(data_tpl['raw_label_file'])
            else:
                filename = basename(data_tpl['raw_image_file'])
                if filename.endswith('_0000.nii.gz'):
                    filename = filename[:-12]

        # remove fileextension e.g. .nii.gz
        filename = filename.split('.')[0] + '.nii.gz'

        # all predictions are stored in the designated 'predictions' folder in the OV_DATA_BASE
        pred_folder = join(environ['OV_DATA_BASE'], 'predictions', self.data_name,
                           self.preprocessed_name, self.model_name, folder_name)
        if not exists(pred_folder):
            makedirs(pred_folder)

        # get storing info from the data_tpl
        # IMPORTANT: We will always store the prediction in original shape
        # not in preprocessed shape
        key = self.pred_key
        if self.pred_key+'_orig_shape' in data_tpl:
            key += '_orig_shape'

        save_nii_from_data_tpl(data_tpl, join(pred_folder, filename), key)

    def plot_prediction(self, data_tpl, folder_name, filename=None, image_key='image'):

        # find name of the file
        if filename is None:
            if 'raw_label_file' in data_tpl:
                filename = basename(data_tpl['raw_label_file'])
            else:
                filename = basename(data_tpl['raw_image_file'])
                if filename.endswith('_0000.nii.gz'):
                    filename = filename[:-12]

        # remove fileextension e.g. .nii.gz
        filename = filename.split('.')[0]

        # all predictions are stored in the designated 'plots' folder in the OV_DATA_BASE
        plot_folder = join(environ['OV_DATA_BASE'], 'plots', self.data_name,
                           self.preprocessed_name, self.model_name, folder_name)
        if not exists(plot_folder):
            makedirs(plot_folder)

        # we want the code to work regardless of wether we have manual segmentaions or not
        # the labels will carry the manual segmentations (in case available) plus the
        # predictions
        labels = []
        im = data_tpl[image_key]
        if torch.is_tensor(im):
            im = im.cpu().numpy()
        if len(im.shape) == 3:
            im = im[np.newaxis]
        im = im.astype(float)
        n_ch = im.shape[0]
        if 'label' in data_tpl:
            # in case of raw data this only removes the lables that this model doesn't segment
            labels.append(self.preprocessing.maybe_clean_label_from_data_tpl(data_tpl))

        labels.append(data_tpl[self.pred_key])
        labels = np.stack(labels)
        # sum over channel, x and y axis
        contains = np.where(np.sum(labels, (0, 2, 3)))[0]
        if len(contains) == 0:
            return

        z_list = [np.argmax(np.sum(labels, (0, 2, 3)))]
        s_list = ['_largest']
        z_list.extend(np.random.choice(contains, size=self.plot_n_random_slices))
        if self.plot_n_random_slices > 1:
            s_list.extend(['_random_{}'.format(i) for i in range(self.plot_n_random_slices)])
        else:
            s_list.append('_random')

        colors = ['r', 'b']
        # now plot largest and random slice
        for z, s in zip(z_list, s_list):
            fig = plt.figure()
            for c in range(n_ch):
                plt.subplot(1, n_ch, c+1)
                plt.imshow(im[c, z], cmap='gray')
                for i in range(labels.shape[0]):
                    if labels[i, z].max() > 0:
                        # this if is purely to avoid annoying UserWarning messages that interrupt
                        # the beautiful beautiful tqdm bar
                        plt.contour(labels[i, ..., z] > 0, linewidths=0.5, colors=colors[i],
                                    linestyles='solid')
                plt.axis('off')
            plt.savefig(join(plot_folder, filename + s + '.png'), bbox_inches='tight')
            plt.close(fig)

    def compute_error_metrics(self, data_tpl):
        if 'label' not in data_tpl:
            # in this case we're evaluating an unlabeled image so we can\'t compute any metrics
            return None
        pred = data_tpl[self.pred_key]
        # in case of raw data this only removes the lables that this model doesn't segment
        seg = self.preprocessing.maybe_clean_label_from_data_tpl(data_tpl)
        results = {}
        for c in range(1, self.n_fg_classes+1):
            seg_c = (seg == c).astype(float)
            pred_c = (pred == c).astype(float)

            has_fg = seg_c.max() > 0
            fg_pred = pred_c.max() > 0

            results.update({'has_fg_%d' % c: seg_c.max() > 0,
                            'fg_pred_%d' % c: fg_pred})
            tp = np.sum(seg_c * pred_c)
            seg_c_vol = np.sum(seg_c)
            pred_c_vol = np.sum(pred_c)
            if has_fg:
                dice = 200 * tp / (seg_c_vol + pred_c_vol)
            else:
                dice = np.nan
            results.update({'dice_%d' % c: dice})

            if has_fg:
                sens = 100 * tp / seg_c_vol
            else:
                sens = np.nan

            if fg_pred:
                prec = 100 * tp / pred_c_vol
            else:
                prec = np.nan

            results.update({'sens_%d' % c: sens, 'prec_%d' % c: prec})
        return results

    def _init_global_metrics(self):
        self.global_metrics_helper = {}
        self.global_metrics = {}
        for c in range(1, self.n_fg_classes + 1):
            self.global_metrics_helper.update({s+str(c): 0 for s in ['overlap_',
                                                                     'gt_volume_',
                                                                     'pred_volume_']})
            self.global_metrics.update({'dice_'+str(c): -1,
                                        'recall_'+str(c): -1,
                                        'precision_'+str(c): -1})

    def _update_global_metrics(self, data_tpl):

        if 'label' not in data_tpl:
            return
        label = data_tpl['label']
        pred = data_tpl[self.pred_key]

        # volume of one voxel
        fac = np.prod(data_tpl['spacing'])
        for c in range(1, self.n_fg_classes + 1):
            lb_c = (label == c).astype(float)
            pred_c = (pred == c).astype(float)
            ovlp = self.global_metrics_helper['overlap_'+str(c)] + np.sum(lb_c * pred_c) * fac
            gt_vol = self.global_metrics_helper['gt_volume_'+str(c)] + np.sum(lb_c) * fac
            pred_vol = self.global_metrics_helper['pred_volume_'+str(c)] + np.sum(pred_c) * fac
            # update global dice, recall and precision
            if gt_vol + pred_vol > 0:
                self.global_metrics['dice_'+str(c)] = 200 * ovlp / (gt_vol + pred_vol)
            else:
                self.global_metrics['dice_'+str(c)] = 100
            if gt_vol > 0:
                self.global_metrics['recall_'+str(c)] = 100 * ovlp / gt_vol
            else:
                self.global_metrics['recall_'+str(c)] = 100 if pred_vol == 0 else 0
            if pred_vol > 0:
                self.global_metrics['precision_'+str(c)] = 100 * ovlp / pred_vol
            else:
                self.global_metrics['precision_'+str(c)] = 100 if gt_vol == 0 else 0

            # now update global metrics helper
            self.global_metrics_helper['overlap_'+str(c)] = ovlp
            self.global_metrics_helper['gt_volume_'+str(c)] = gt_vol
            self.global_metrics_helper['pred_volume_'+str(c)] = pred_vol

    def preprocess_prediction_for_next_stage(self, prep_name_next_stage):

        print('Preprocessing cross validation predictions for the next stage...\n\n')
        # first check if all the prediction from this model are actually there
        pred_folder = join(environ['OV_DATA_BASE'], 'predictions', self.data_name,
                           self.preprocessed_name, self.model_name, 'cross_validation')

        cases_missing = False
        print('Checking if all predictions are there.')
        if exists(pred_folder):
            for scan in self.data.val_ds.used_scans:
                nii_file = basename(scan).split('.')[0] + '.nii.gz'
                if nii_file not in listdir(pred_folder):
                    cases_missing = True
        else:
            cases_missing = True

        if cases_missing:
            print('Not all validation cases were found in the prediction path '+pred_folder)
            print('Doing the validation prediction now.\n')
            self.eval_validation_set(save_preds=True, save_plots=False, force_evaluation=True)

        prep_folder_next_stage = join(environ['OV_DATA_BASE'], 'preprocessed', self.data_name,
                                      prep_name_next_stage)
        prep_pred_folder = join(prep_folder_next_stage,
                                self.preprocessed_name + '_' + self.model_name)
        if not exists(prep_pred_folder):
            makedirs(prep_pred_folder)
        # pickled paramters used for the next stage
        prep_params_next_stage = load_pkl(join(prep_folder_next_stage,
                                               'preprocessing_parameters.pkl'))

        # create the preprocessing object for the next stage
        # IMPORTANT! This is not the same preprocessing object as
        #   self.preprocessing_for_pred_from_prev_stage
        # this one is being used to preprocess prediction from a previous, to this stage
        params_ps = {'apply_windowing': False,
                     'scaling': [1, 0],
                     'apply_resizing': prep_params_next_stage['apply_resizing'],
                     'apply_pooling': prep_params_next_stage['apply_pooling'],
                     'do_nn_img_interp': True}
        if params_ps['apply_resizing']:
            params_ps['target_spacing'] = prep_params_next_stage['target_spacing']
        if params_ps['apply_pooling']:
            params_ps['pooling_stride'] = prep_params_next_stage['pooling_stride']

        print('Creating preprocessing object for next stage...')
        preprocessing_for_next_stage = SegmentationPreprocessing(**params_ps)

        # cool! Let's go on and cylce through the cases
        print('Preprocessing nifti predictions for next stage')
        sleep(1)
        for scan in tqdm(self.data.val_ds.used_scans):
            scan_name = basename(scan).split('.')[0]
            nii_file = join(pred_folder, scan_name + '.nii.gz')
            im, spacing, _ = read_nii(nii_file)
            lb_prep = preprocessing_for_next_stage({'image': im, 'spacing': spacing},
                                                   return_np=True)
            np.save(join(prep_pred_folder, scan_name+'.npy'), lb_prep[0].astype(np.uint8))

        print('Done!')
        return
