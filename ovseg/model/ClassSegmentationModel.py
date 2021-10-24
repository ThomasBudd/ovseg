from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.preprocessing.ClassSegmentationPreprocessing import ClassSegmentationPreprocessing
from ovseg.data.ClassSegmentationData import ClassSegmentationData
from ovseg.utils.torch_np_utils import maybe_add_channel_dim
import numpy as np

class ClassSegmentationModel(SegmentationModel):

    
    def _create_preprocessing_object(self):
        
        self.preprocessing = ClassSegmentationPreprocessing(**self.model_parameters['preprocessing'])

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
        self.data = ClassSegmentationData(val_fold=self.val_fold,
                                          preprocessed_path=self.preprocessed_path,
                                          augmentation= self.augmentation.np_augmentation,
                                          **params)
        print('Data initialised')


    def initialise_training(self):
        if 'training' not in self.model_parameters:
            raise AttributeError('model_parameters must have key '
                                 '\'training\'. These must contain the '
                                 'dict of training paramters.')
        if 'batches_have_masks' in self.model_parameters['training']:
            if not self.model_parameters['training']['batches_have_masks']:
                print('batches_have_masks was set to False, but should always be true')
        else:
            self.model_parameters['training']['batches_have_masks'] = True
            
            if self.parameters_match_saved_ones:
                print('Added \'batches_have_masks\'=True to training params. Saving paramters')
                self.save_model_parameters()

        super().initialise_training()

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
            vol = self.preprocessing(data_tpl, preprocess_only_im=True)
            im, pp = vol, vol[-1:]
        else:
            # the data_tpl is already preprocessed, let's just get the arrays
            im = maybe_add_channel_dim(data_tpl['image'])
            pp = maybe_add_channel_dim(data_tpl['prev_pred'])
            im = np.concatenate([im, pp])

        # now the importat part: the sliding window evaluation (or derivatives of it)
        pred = self.prediction(im, pp[0]>0)
        data_tpl[self.pred_key] = pred

        # inside the postprocessing the result will be attached to the data_tpl
        if do_postprocessing:
            self.postprocessing.postprocess_data_tpl(data_tpl, self.pred_key, pp)

        return data_tpl[self.pred_key]
