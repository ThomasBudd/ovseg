import warnings
warnings.simplefilter("ignore")
import torch
import numpy as np
from os import environ, makedirs, listdir
from os.path import join, basename, exists
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = lambda x: x
from time import sleep
from skimage.measure import label
from ovseg import OV_PREPROCESSED
from ovseg.model.ModelBase import ModelBase
from ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
from ovseg.augmentation.SegmentationAugmentation import SegmentationAugmentation
from ovseg.data.SegmentationData import SegmentationData
from ovseg.data.Dataset import raw_Dataset
from ovseg.networks import __dict__ as networks_dict
from ovseg.training.SegmentationTraining import SegmentationTraining, SegmentationTrainingV2
from ovseg.prediction.SlidingWindowPrediction import SlidingWindowPrediction
from ovseg.postprocessing.SegmentationPostprocessing import SegmentationPostprocessing
from ovseg.utils.io import save_nii, load_pkl, read_nii, save_dcmrt_from_data_tpl, is_dcm_path
from ovseg.utils.torch_np_utils import maybe_add_channel_dim
from ovseg.utils.dict_equal import dict_equal, print_dict_diff
from ovseg.utils.label_utils import reduce_classes


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

        if self.is_cascade():
            self.prev_stages = self.model_parameters['prev_stages']
            self.prev_stages_keys = []
            for prev_stage in self.prev_stages:
                key = '_'.join(['prediction',
                                prev_stage['data_name'],
                                prev_stage['preprocessed_name'],
                                prev_stage['model_name']])
                self.prev_stages_keys.append(key)


    def is_cascade(self):
        if 'prev_stages' in self.model_parameters:
            if self.model_parameters['prev_stages'] is None:
                return False
            elif isinstance(self.model_parameters['prev_stages'], list):
                return len(self.model_parameters['prev_stages']) > 0
            else:
                raise TypeError('Got type {} for prev_stages, expected list or none.'
                                ''.format(type(self.model_parameters['prev_stages'])))
        else:
            return False

    def _create_preprocessing_object(self):
        
        self.preprocessing = SegmentationPreprocessing(**self.model_parameters['preprocessing'])

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
                if 'prev_stages' in prep_params:
                    self.model_parameters['prev_stages'] = prep_params['prev_stages']
                if self.parameters_match_saved_ones:
                    print('Loaded preprocessing parameters and updating model '
                          'parameters.')
                    self.save_model_parameters()
                else:
                    print('Loaded preprocessing parameters without saving them to the model '
                          'parameters as current model parameters don\'t match saved ones.')

        params = self.model_parameters['preprocessing'].copy()
        if self.is_cascade():
            # Let's compare if the prev stages in the model and preprocessing paramters match
            # OK this is ugly, but dict_equal can not compare lists...
            d1 = {'prev_stages': self.model_parameters['prev_stages']}
            d2 = {'prev_stages': params['prev_stages']}
            if not dict_equal(d1, d2):
                print_dict_diff(d1, d2, 'model_parameters', 'preprocessing_paramters')
                raise ValueError('Found missmatch between prev stages given in the model paramters '
                                 'and the preprocessing parameters!')
    
        self._create_preprocessing_object()

        # now for the computation of loss metrics we need the number of prevalent fg classes
        if self.preprocessing.reduce_lb_to_single_class:
            self.n_fg_classes = 1
        elif self.preprocessing.lb_classes is not None:
            self.n_fg_classes = len(self.preprocessing.lb_classes)
        elif self.model_parameters['network']['out_channels'] is not None:
            self.n_fg_classes = self.model_parameters['network']['out_channels'] - 1
        elif hasattr(self.preprocessing, 'dataset_properties'):
            print('Using all foreground classes for computing the DSCS')
            self.n_fg_classes = self.preprocessing.dataset_properties['n_fg_classes']
        else:
            raise AttributeError('Something seems to be wrong. Could not figure out the number '
                                 'of foreground classes in the problem...')
        if self.preprocessing.lb_classes is None and hasattr(self.preprocessing, 'dataset_properties'):
            
            if self.preprocessing.reduce_lb_to_single_class:
                self.lb_classes = [1]
            else:
                self.lb_classes = list(range(1, self.n_fg_classes+1))
                if self.n_fg_classes != self.preprocessing.dataset_properties['n_fg_classes']:
                    print('There seems to be a missmatch between the number of forground '
                          'classes in the preprocessed data and the number of network '
                          'output channels....')
        else:
            self.lb_classes = self.preprocessing.lb_classes

    def initialise_augmentation(self):

        # the augmentation object carries two subobjects, one with the preprocessing
        # happning in numpy on the CPU and on in torch on the GPU
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
        # wow yes this is super ugly! It would be better to include all architecture
        # in one file and the use
        # from FILE import __dict__
        # self.network = __dict__[self.model_paramters['architecture']]
        arc = self.model_parameters['architecture']
        params = self.model_parameters['network']
        for key in networks_dict:
            # check case insensitive for matching architecture
            if key.lower() == arc.lower():
                self.network = networks_dict[key](**params).to(self.dev)
        
        if not hasattr(self, 'network'):
            raise ValueError(f'Architecture {arc} not found.'
                             'Make sure it is imported in ovseg.network.__init__.py')
        
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
            params = {}
        # the SegmentationPostprocessing is relatively uninteresting, what happens here
        # is the resizing to the original volume, applying argmax, maybe removing some small
        # connected components
        params.update({'lb_classes': self.preprocessing.lb_classes})
        
        self.postprocessing = SegmentationPostprocessing(**params)

    def initialise_data(self):
        # the data object holds the preprocessed data (training and validation)
        # for each it has both a dataset returning the data tuples and the dataloaders
        # returning the batches
        if 'data' not in self.model_parameters:
            raise AttributeError('model_parameters must have key '
                                 '\'data\'. These must contain the '
                                 'dict of training paramters.')

        # Let's get the parameters and add the cpu augmentation
        params = self.model_parameters['data'].copy()

        # if we don't want to store our data in ram...
        if self.dont_store_data_in_ram:
            for key in ['trn_dl_params', 'val_dl_params']:
                params[key]['store_data_in_ram'] = False
                params[key]['store_coords_in_ram'] = False
        self.data = SegmentationData(val_fold=self.val_fold,
                                     preprocessed_path=self.preprocessed_path,
                                     augmentation= self.augmentation.np_augmentation,
                                     **params)
        print('Data initialised')

    def initialise_training(self):
        # the magic! The training takes in a lot of things we've already initialised and
        # takes care of the training and making the logs
        if 'training' not in self.model_parameters:
            raise AttributeError('model_parameters must have key '
                                 '\'training\'. These must contain the '
                                 'dict of training paramters.')
        params = self.model_parameters['training'].copy()
        try:
            self.training = SegmentationTrainingV2(network=self.network,
                                                   trn_dl=self.data.trn_dl,
                                                   val_dl=self.data.val_dl,
                                                   model_path=self.model_path,
                                                   network_name=self.network_name,
                                                   augmentation=self.augmentation.torch_augmentation,
                                                   **params)
        except TypeError:
            print('Caught a Type error, are you trying to load an older model? '
                  'Creating training object with old implementation of loss function.')
            self.training = SegmentationTraining(network=self.network,
                                                 trn_dl=self.data.trn_dl,
                                                 val_dl=self.data.val_dl,
                                                 model_path=self.model_path,
                                                 network_name=self.network_name,
                                                 augmentation=self.augmentation.torch_augmentation,
                                                 **params)

    def __call__(self, data_tpl, do_postprocessing=True):
        '''
        This function just predict the segmentation for the given data tpl
        There are a lot of differnt ways to do prediction. Some do require direct preprocessing
        some don't need the postprocessing imidiately (e.g. when ensembling)
        Same holds for the resizing to original shape. In the validation case we wan't to apply
        some postprocessing (argmax and removing of small lesions) but not the resizing.
        '''
        self.network = self.network.eval()

        # first let's get the image and maybe the bin_pred as well
        # the preprocessing will only do something if the image is not preprocessed yet
        if not self.preprocessing.is_preprocessed_data_tpl(data_tpl):
            # the image already contains the binary prediction as additional channel
            im = self.preprocessing(data_tpl, preprocess_only_im=True)
            if self.is_cascade():
                bin_pred = im[-1:]
            else:
                bin_pred = None
        else:
            # the data_tpl is already preprocessed, let's just get the arrays
            im = data_tpl['image']
            im = maybe_add_channel_dim(im)
            if self.is_cascade():
                # if the data tpl is preprocessed we need to build the binary prediction here
                # prev_preds = []
                # for key in self.prev_stages_s:
                #     assert key in data_tpl, 'prediction '+key+' from previous stage missing'
                #     pred = data_tpl[key]
                #     if torch.is_tensor(pred):
                #         pred = pred.cpu().numpy()
                #     if len(pred.shape) == 3:
                #         pred = pred[np.newaxis]
                #     prev_preds.append(pred)
        
                # bin_pred = (np.sum(prev_preds, 0) > 0).astype(float)
                # if the data_tpl is preprocessed the binary prediction should already be in there
                bin_pred = data_tpl['bin_pred'][np.newaxis]
                im = np.concatenate([im, bin_pred])
            else:
                bin_pred = None
        # now the importat part: the sliding window evaluation (or derivatives of it)
        pred = self.prediction(im)
        data_tpl[self.pred_key] = pred

        # inside the postprocessing the result will be attached to the data_tpl
        if do_postprocessing:
            self.postprocessing.postprocess_data_tpl(data_tpl, self.pred_key, bin_pred)

        return data_tpl[self.pred_key]

    def save_prediction(self, data_tpl, folder_name, filename=None):

        # find name of the file
        if filename is None:
            filename = data_tpl['scan'] + '.nii.gz'
            out_file = data_tpl['scan'] + '.dcm'
        else:
            # remove fileextension e.g. .nii.gz
            filename = filename.split('.')[0] + '.nii.gz'
            out_file = filename.split('.')[0] + '.dcm'

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
        
        rif = data_tpl['raw_image_file']
        if rif.endswith('.nii') or rif.endswith('.nii.gz'):
            save_nii(data_tpl[key], 
                     join(pred_folder, filename), 
                     rif)
        
        if is_dcm_path(data_tpl['raw_image_file']):
            
            red_key = key+'dcm_export'
            data_tpl[red_key] = reduce_classes(data_tpl[key], self.lb_classes)
            names = [str(lb) for lb in self.lb_classes]
            save_dcmrt_from_data_tpl(data_tpl, join(pred_folder, out_file),
                                     key=red_key, names=names)

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
        im = maybe_add_channel_dim(im).astype(float)
        n_ch = im.shape[0]
        
        pred = data_tpl[self.pred_key]
        pred = maybe_add_channel_dim(pred)
        
        labels.append(pred)
        if 'label' in data_tpl:
            # in case of raw data this only removes the lables that this model doesn't segment
            lb = self.preprocessing.maybe_clean_label_from_data_tpl(data_tpl)
            lb = maybe_add_channel_dim(lb)
            labels.append(lb)

        labels = np.concatenate(labels)
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
        # seg = self.preprocessing.maybe_clean_label_from_data_tpl(data_tpl)
        # with the new update the prediction should be in classes as well instead of 
        # integer encoding as before. Let's hope that it works!
        seg = data_tpl['label']
        if len(seg.shape) == 4:
            seg = seg[0]

        if self.preprocessing.is_preprocessed_data_tpl(data_tpl):
            # if we have a preprocessed data_tpl we need to bring the segmentation back from
            # integer to class encoding
            seg_lb = np.zeros_like(seg)
            for i, c in enumerate(self.lb_classes):
                seg_lb[seg == i+1] = c
            seg = seg_lb
        else:
            
            # remove all uninteresting lb classes fr bin_dice computation
            seg_lb = np.zeros_like(seg)
            for c in self.lb_classes:
                seg_lb[seg == c] = c
            seg = seg_lb
            if self.preprocessing.reduce_lb_to_single_class:
                seg = (seg > 0).astype(seg.dtype)
        # results are returned as a dict
        results = {}
        
        # compute the bin dsc for multiple classes
        if len(self.lb_classes) > 1:
            bin_seg = (seg > 0).astype(float)
            bin_pred = (pred > 0).astype(float)
            results['bin_dice'] = 200 * np.sum(bin_seg * bin_pred) /  \
                    np.sum(bin_seg + bin_pred)

        for c in self.lb_classes:
            seg_c = (seg == c).astype(float)
            pred_c = (pred == c).astype(float)

            # has_fg = seg_c.max() > 0
            # fg_pred = pred_c.max() > 0

            # results.update({'has_fg_%d' % c: has_fg,
            #                 'fg_pred_%d' % c: fg_pred})
            tp = np.sum(seg_c * pred_c)
            seg_c_vol = np.sum(seg_c)
            pred_c_vol = np.sum(pred_c)
            if seg_c_vol > 0:
                dice = 200 * tp / (seg_c_vol + pred_c_vol)
                sens = 100 * tp / seg_c_vol
            else:
                dice = np.nan
                sens = np.nan
            results.update({'dice_%d' % c: dice, 'sens_%d' % c: sens})

            if pred_c_vol > 0 and seg_c_vol > 0:
                prec = 100 * tp / pred_c_vol
            else:
                prec = np.nan

            results.update({'prec_%d' % c: prec})

        if len(self.lb_classes) > 1:
            dscs = [results['dice_%d'%c] for c in self.lb_classes]
            results['dice_mc'] = np.mean(dscs)

        return results

    def _init_global_metrics(self):
        self.global_metrics = {}
        for c in self.lb_classes:#range(1, self.n_fg_classes + 1):
            self.global_metrics.update({s+str(c): 0 for s in ['_overlap_',
                                                              '_gt_volume_',
                                                              '_pred_volume_']})
            self.global_metrics.update({'dice_'+str(c): -1,
                                        'recall_'+str(c): -1,
                                        'precision_'+str(c): -1})

    def _update_global_metrics(self, data_tpl):

        if 'label' not in data_tpl:
            return
        
        pred = data_tpl[self.pred_key]
        seg = data_tpl['label']
        if len(seg.shape) == 4:
            seg = seg[0]

        if self.preprocessing.is_preprocessed_data_tpl(data_tpl):
            # if we have a preprocessed data_tpl we need to bring the segmentation back from
            # integer to class encoding
            seg_lb = np.zeros_like(seg)
            for i, c in enumerate(self.lb_classes):
                seg_lb[seg == i+1] = c
            seg = seg_lb
        else:
            if self.preprocessing.reduce_lb_to_single_class:
                seg = (seg > 0).astype(seg.dtype)

        # volume of one voxel
        fac = np.prod(data_tpl['spacing']) if 'spacing' in data_tpl else 1
        for c in self.lb_classes:#range(1, self.n_fg_classes + 1):
            lb_c = (seg == c).astype(float)
            pred_c = (pred == c).astype(float)
            ovlp = self.global_metrics['_overlap_'+str(c)] + np.sum(lb_c * pred_c) * fac
            gt_vol = self.global_metrics['_gt_volume_'+str(c)] + np.sum(lb_c) * fac
            pred_vol = self.global_metrics['_pred_volume_'+str(c)] + np.sum(pred_c) * fac
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

            # now update global metrics helpers
            self.global_metrics['_overlap_'+str(c)] = ovlp
            self.global_metrics['_gt_volume_'+str(c)] = gt_vol
            self.global_metrics['_pred_volume_'+str(c)] = pred_vol
    
    def _merge_global_results_to_CV(self, ds_name):
        
        # reset global metrics
        self._init_global_metrics()
        
        # load globale results from all folds
        all_folds = [fold for fold in listdir(self.model_cv_path) if fold.startswith('fold')]
        for fold in all_folds:
            if ds_name+'_results_per_scan.pkl' in listdir(join(self.model_cv_path, fold)):
                fold_results = load_pkl(join(self.model_cv_path, fold,
                                             ds_name+'_results_global.pkl'))
                
                # update overlap and volumes
                for key in fold_results:
                    if key.startswith('_'):
                        self.global_metrics[key] += fold_results[key]
            
        # now update DSC, SENS and PREC
        for c in self.lb_classes:
            ovlp = self.global_metrics['_overlap_'+str(c)]
            gt_vol = self.global_metrics['_gt_volume_'+str(c)] 
            pred_vol = self.global_metrics['_pred_volume_'+str(c)] 
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

    def preprocess_prediction_for_next_stage(self, prep_name_next_stage):

        if not hasattr(self.data, 'val_ds'):
            print('Model has no validation data. There is nothing to preprocess for the next '
                  'stage. Exeting!')
            return

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

        prep_folder_next_stage = join(OV_PREPROCESSED, self.data_name,
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

    def clean(self):
        # deletes (hopefully) all data from ram and the network from the GPU
        if hasattr(self, 'data'):
            self.data.clean()
        
        self.network = self.network.cpu()
        del self.network
        torch.cuda.empty_cache()

    def eval_raw_data_npz(self, raw_data_name,
                          scans=None, image_folder=None, dcm_revers=True,
                          dcm_names_dict=None):
        # this function predicts the images and raw data and saves the 
        # predictions before thresholding. This is usefull for ensembling when
        # the prediction takes time. This way all models in the ensemble can run the prediction
        # indepentently and the ensemble just has to collect the results --> multi GPU ensembling
        ds = raw_Dataset(join(environ['OV_DATA_BASE'], 'raw_data', raw_data_name),
                         scans=scans,
                         image_folder=image_folder,
                         dcm_revers=dcm_revers,
                         dcm_names_dict=dcm_names_dict,
                         prev_stages=self.prev_stages if hasattr(self, 'prev_stages') else None)

        if len(ds) == 0:
            print('Got empty dataset for evaluation. Nothing to do here --> leaving!')
            return

        # we have a destinct folder for the npz predictions. As they take a lot of disk space
        # this makes it easier to delete them
        pred_npz_path = join(environ['OV_DATA_BASE'], 'npz_predictions', self.data_name,
                             self.preprocessed_name, self.model_name, self.val_fold_str)

        if not exists(pred_npz_path):
            makedirs(pred_npz_path)

        print('Evaluating '+raw_data_name+' '+self.val_fold_str+'...\n\n')
        sleep(1)
        for i in tqdm(range(len(ds))):
            # get the data
            data_tpl = ds[i]
            # first let's try to find the name
            scan = data_tpl['scan']
            if exists(join(pred_npz_path, scan+'.npz')) or exists(join(pred_npz_path, scan+'.npy')):
                continue
            # now let's do (almost the full) prediction
            pred = self.__call__(data_tpl, do_postprocessing=False)
            if torch.is_tensor(pred):
                pred = pred.cpu().numpy()
            pred = pred.astype(np.float16)
            # np.savez_compressed(join(pred_npz_path, scan), pred)
            np.save(join(pred_npz_path, scan), pred)

    def infere_volume_thresholds(self, folder_name='cross_validation', scans=None,
                                 image_folder=None, dcm_revers=True, dcm_names_dict=None):
        # raw dataset to access ground truth data
        ds = raw_Dataset(join(environ['OV_DATA_BASE'], 'raw_data', self.data_name),
                         scans=scans,
                         image_folder=image_folder,
                         dcm_revers=dcm_revers,
                         dcm_names_dict=dcm_names_dict)
        # path with predictions (should be stored as nibabel)
        predp = join(environ['OV_DATA_BASE'], 'predictions', self.data_name, self.preprocessed_name,
                     self.model_name, folder_name)
        if self.n_fg_classes > 1:
            print('WARNING: finding optimal volume treshold is atm only implemented for '
                  'single class problems.')
        vols_delta_dsc = []
        for i in tqdm(range(len(ds))):
            data_tpl = ds[i]
            # get ground truth and possible remove other labels from the image
            gt = (self.preprocessing.maybe_clean_label_from_data_tpl(data_tpl) > 0).astype(float)
            pred = read_nii(join(predp, data_tpl['scan']+'.nii.gz'))[0] > 0
            # all connected components, but how to set the threshold for removing the too small ones?
            comps = label(pred)

            # here we collect both the volume of the component as well as the
            # arrays containing just the component
            vols_and_comps = [(np.sum(comps == c), comps == c) for c in range(1, comps.max() + 1)]
            # sort the list to start with the smallest volume
            vols_and_comps = sorted(vols_and_comps)

            # this choice doesn't matter, will shift the total stats only by a constant
            dsc_old = 0
            for j in range(len(vols_and_comps)):
                # prepare to fill in all components that are greater than the current threshold
                pred_tr = np.zeros_like(pred, dtype=float)
                for _, comp in vols_and_comps[j+1:]:
                    pred_tr[comp] = 1
                dsc_new = 200*np.sum(pred_tr * gt) / np.sum(gt + pred_tr)
                vols_delta_dsc.append(vols_and_comps[j][0], dsc_new - dsc_old)
                dsc_old = dsc_new
 