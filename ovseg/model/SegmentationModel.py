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
from ovseg.utils.io import save_nii, save_pkl, load_pkl
from ovseg.utils.eval_predictions import eval_prediction_segmentation
from torch.nn import functional as F
import torch
import numpy as np
from os import listdir, mkdir
from os.path import join, exists, basename
import matplotlib.pyplot as plt
from tqdm import tqdm


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

        if 'prediction_key' not in self.model_parameters:
            model_parameters['prediction_key'] = 'learned_segmentation'
            print('\'prediction_key\' not initialised, set to \'learned_segmentation\'.')

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
                self.model_parameters.update({'preprocessing': prep_params})
                if self.parameters_match_saved_ones:
                    print('Loaded preprocessing parameters and updating model '
                          'parameters.')
                    self.save_model_parameters()
                    self._model_parameters_to_txt()
                else:
                    print('Loaded preprocessing parameters without saving them to the model '
                          'parameters as current model parameters don\'t match saved ones.')
        params = self.model_parameters['preprocessing'].copy()
        self.preprocessing = SegmentationPreprocessing(**params)
        if self.preprocessing.use_only_classes is not None:
            self.n_fg_classes = len(self.preprocessing.use_only_classes)
        else:
            try:
                self.n_fg_classes = self.preprocessing.dataset_properties.n_fg_classes
            except AttributeError:
                print('Could not find number of fg classes appearing in the data.\n'
                      'This will cause a problem when computing evaluation metrics.\n'
                      'Setting the n_fg_classes to 1 (computing only DSC etc. of first class).')
                self.n_fg_classes = 1
        print('Preprocessing initialised.\n')

    def initialise_augmentation(self):

        # first initialise CPU augmentation
        # this happens in the dataloader
        if 'augmentation' in self.model_parameters:
            self.augmentation = SegmentationAugmentation(**self.model_parameters['augmentation'])
        print('Augmentation initialised')

    def initialise_network(self):
        if 'network' not in self.model_parameters:
            raise AttributeError('model_parameters must have key '
                                 '\'network\'. These must contain the '
                                 'dict of network paramters.')
        params = self.model_parameters['network'].copy()
        if self.model_parameters['architecture'].lower() in ['unet', 'u-net']:
            self.network = UNet(**params).cuda()
        elif self.model_parameters['architecture'].lower() in ['iunet', 'iu-net']:
            raise NotImplementedError('CHRISTIAN!!! CHRISTIAN!!! Come and do this.')
            # self.network = iUNet(**params)
        print('Network initialised')

    def initialise_prediction(self):
        params = {'network': self.network,
                  'patch_size': self.model_parameters['data']['trn_dl_params']['patch_size']}
        if 'prediction' not in self.model_parameters:
            print('model_parameters doesn\'t have key \'prediction\' to speficfy how full volumes '
                  'are processed by the model. Using default parameters')
        else:
            params.update(self.model_parameters['prediction'])

        self.prediction = SlidingWindowPrediction(**params, TTA=self.augmentation.TTA)

    def initialise_postprocessing(self):
        try:
            params = self.model_parameters['postprocessing'].copy()
        except KeyError:
            print('No parameter for postprocessing were given. Take defaut '
                  'values (no removing of small connected components.')
            params = {}
        self.postprocessing = SegmentationPostprocessing(**params)
        print('Postprocessing initialised')

    def initialise_data(self):
        if 'data' not in self.model_parameters:
            raise AttributeError('model_parameters must have key '
                                 '\'data\'. These must contain the '
                                 'dict of training paramters.')

        # Let's get the parameters and add the cpu augmentation
        params = self.model_parameters['data'].copy()
        try:
            data_aug = self.augmentation.CPU_augmentation
        except AttributeError:
            data_aug = None

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
        self.training = \
            SegmentationTraining(network=self.network,
                                 trn_dl=self.data.trn_dl,
                                 val_dl=self.data.val_dl,
                                 model_path=self.model_path,
                                 network_name=self.network_name,
                                 augmentation=self.augmentation.GPU_augmentation,
                                 **params)
        print('Training initialised')

    def predict(self, data_tpl, is_preprocessed):
        '''
        There are a lot of differnt ways to do prediction. Some do require direct preprocessing
        some don't need the postprocessing imidiately (e.g. when ensembling)
        Same holds for the resizing to original shape. In the validation case we wan't to apply
        some postprocessing (argmax and removing of small lesions) but not the resizing.
        '''
        self.network = self.network.eval()
        im = data_tpl['image']
        is_np,  _ = check_type(im)
        if is_np:
            im = torch.from_numpy(im).cuda()
        else:
            im = im.cuda()

        # most of the time we expect this to be False
        # Validation scans should be prepocessed and when we perform ensembling to testing
        # we would only preprocess the data once and then hand these images in each model
        with torch.no_grad():
            if not is_preprocessed:
                orig_shape = im.shape
                im = self.preprocessing(im, data_tpl['spacing'])
            else:
                orig_shape = None

            # now the importat part: the sliding window evaluation (or derivatices of it)
            pred = self.prediction(im)

            pred = self.postprocessing(pred, orig_shape)

        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()

        data_tpl[self.model_parameters['prediction_key']] = pred

        return pred

    def save_prediction(self, pred, data, pred_folder, name):

        out_file = join(pred_folder, basename(data['raw_label_file']))

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

        contains = np.where(np.sum((seg + pred) > 0, (0, 1)))[0]
        if len(contains) == 0:
            return

        z_list = []
        s_list = []
        if seg.max() > 0:
            z_list.append(np.argmax(np.sum(seg, (0, 1))))
            s_list.append('_largest')
        z_list.extend(np.random.choice(contains, size=self.plot_n_random_slices))
        if self.plot_n_random_slices > 1:
            s_list.extend(['_random_{}'.format(i) for i in range(self.plot_n_random_slices)])
        else:
            s_list.append('_random')

        # now plot largest and random slice
        for z, s in zip(z_list, s_list):
            fig = plt.figure()
            plt.imshow(im[..., z], cmap='gray')
            plt.contour(seg[..., z] > 0, linewidths=0.5, colors='r', linestyles='solid')
            plt.contour(pred[..., z] > 0, linewidths=0.5, colors='b', linestyles='solid')
            plt.axis('off')
            plt.savefig(join(plot_folder, name + s + '.png'))
            plt.close(fig)

    def compute_error_metrics(self, pred, seg):
        results = {}
        for c in range(1, self.n_fg_classes+1):
            results = {}
            seg_c = (seg == c).astype(float)
            pred_c = (pred == c).astype(float)

            has_fg = seg_c.max() > 0
            fg_pred = pred_c.max() > 0

            results.update({'has_fg_%d' % c: seg_c.max() > 0,
                            'fg_pred_%d' % c: fg_pred})
            tp = np.sum(seg_c * pred_c)
            seg_c_vol = np.sum(seg_c)
            pred_c_vol = np.sum(pred_c)
            if has_fg or fg_pred:
                dice = 200 * tp / (seg_c_vol + pred_c_vol)
            else:
                dice = 100
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
