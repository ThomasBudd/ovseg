from ovseg.model.ClassSegmentationModel import ClassSegmentationModel
from ovseg.utils.torch_np_utils import maybe_add_channel_dim
import numpy as np


class ClassSegmentationModelV2(ClassSegmentationModel):

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
        else:
            # the data_tpl is already preprocessed, let's just get the arrays
            im = maybe_add_channel_dim(data_tpl['image'])
            pp = maybe_add_channel_dim(data_tpl['prev_pred'])
            im = np.concatenate([im, pp])

        # now the importat part: the sliding window evaluation (or derivatives of it)
        pred = self.prediction(im)
        data_tpl[self.pred_key] = pred

        # inside the postprocessing the result will be attached to the data_tpl
        if do_postprocessing:
            self.postprocessing.postprocess_data_tpl(data_tpl, self.pred_key)

        return data_tpl[self.pred_key]