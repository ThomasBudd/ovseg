from ovseg.postprocessing import SegmentationPostprocessing

class ClassEnsemblingPostprocessing(SegmentationPostprocessing):
    
    def postprocess_data_tpl(self, data_tpl, prediction_key, bin_pred):

        data_tpl = super().postprocess_data_tpl(data_tpl, prediction_key)

        data_tpl[prediction_key]