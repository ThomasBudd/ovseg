from ovseg.model.SegmentationModel import SegmentationModel    
from ovseg.postprocessing.ClassCascadePostprocessing import ClassCascadePostprocessing
from ovseg.preprocessing.ClassCascadePreprocessing import ClassCascadePreprocessing
import numpy as np
from ovseg.utils.torch_np_utils import maybe_add_channel_dim
from ovseg.data.ClassCascadeData import ClassCascadeData


class ClassCascadeModel(SegmentationModel):


    def _create_preprocessing_object(self):
        
        self.preprocessing = ClassCascadePreprocessing(**self.model_parameters['preprocessing'])    

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
        self.data = ClassCascadeData(val_fold=self.val_fold,
                                     preprocessed_path=self.preprocessed_path,
                                     augmentation= self.augmentation.np_augmentation,
                                     **params)
        print('Data initialised')    

    def initialise_postprocessing(self):
        try:
            params = self.model_parameters['postprocessing'].copy()
        except KeyError:
            params = {}
        # the SegmentationPostprocessing is relatively uninteresting, what happens here
        # is the resizing to the original volume, applying argmax, maybe removing some small
        # connected components
        params.update({'lb_classes': self.preprocessing.lb_classes})
        
        self.postprocessing = ClassCascadePostprocessing(**params)

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
            volume = self.preprocessing(data_tpl, preprocess_only_im=True, return_np=True)
            im, prev_pred = volume[:-1], volume[-1:]
        else:
            # the data_tpl is already preprocessed, let's just get the arrays
            im = maybe_add_channel_dim(data_tpl['image'])
            prev_pred = maybe_add_channel_dim(data_tpl['prev_pred'])

        # the input to the network (predicion object) must be with binarised
        # previous prediction
        # im = np.concatenate([im, (prev_pred > 0).astype(im.dtype)])
        # now the importat part: the sliding window evaluation (or derivatives of it)
        pred = self.prediction(im)
        data_tpl[self.pred_key] = pred

        # inside the postprocessing the result will be attached to the data_tpl
        # now for the postprocessing we need to input the non binary previous
        # prediction
        if do_postprocessing:
            self.postprocessing.postprocess_data_tpl(data_tpl, self.pred_key, prev_pred)

        return data_tpl[self.pred_key]
