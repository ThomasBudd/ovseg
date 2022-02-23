from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.preprocessing.SegmentationPreprocessingV2 import SegmentationPreprocessingV2
from ovseg.data.SegmentationDataV2 import SegmentationDataV2
from ovseg.utils.io import save_nii_from_data_tpl, save_npy_from_data_tpl, load_pkl, read_nii, save_dcmrt_from_data_tpl, is_dcm_path
from ovseg.utils.torch_np_utils import maybe_add_channel_dim
from ovseg.utils.dict_equal import dict_equal, print_dict_diff
from os.path import join
import numpy as np

class SegmentationModelV2(SegmentationModel):
    
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
                
                
        params = self.model_parameters['preprocessing'].copy()
    
        self.preprocessing = SegmentationPreprocessingV2(**params)

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
        self.data = SegmentationDataV2(val_fold=self.val_fold,
                                       preprocessed_path=self.preprocessed_path,
                                       augmentation= self.augmentation.np_augmentation,
                                       **params)
        print('Data initialised')


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
            if self.preprocessing.has_ps_mask:
                im, mask = im[:-1], im[-1:]
            else:
                mask = None
        else:
            # the data_tpl is already preprocessed, let's just get the arrays
            im = data_tpl['image']
            im = maybe_add_channel_dim(im)
            if self.preprocessing.has_ps_input:
                
                pred = maybe_add_channel_dim(data_tpl['prev_pred'])
                if pred.max() > 1:
                    raise NotImplementedError('Didn\'t implement the casacde for multiclass'
                                              'prev stages. Add one hot encoding.')
                im = np.concatenate([im, pred])
            
            if self.preprocessing.has_ps_mask:
                mask = maybe_add_channel_dim(data_tpl['mask'])
            else:
                mask = None
            
            
        # now the importat part: the sliding window evaluation (or derivatives of it)
        pred = self.prediction(im)
        data_tpl[self.pred_key] = pred

        # inside the postprocessing the result will be attached to the data_tpl
        if do_postprocessing:
            self.postprocessing.postprocess_data_tpl(data_tpl, self.pred_key, mask)

        return data_tpl[self.pred_key]