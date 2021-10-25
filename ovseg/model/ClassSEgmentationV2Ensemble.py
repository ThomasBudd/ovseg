from ovseg.model.SegmentationEnsemble import SegmentationEnsemble
from ovseg.model.ClassSegmentationModelV2 import ClassSegmentationModelV2


class ClassSegmentationV2Ensemble(SegmentationEnsemble):
    
    def create_model(self, fold):
        model = ClassSegmentationModelV2(val_fold=fold,
                                         data_name=self.data_name,
                                         model_name=self.model_name,
                                         model_parameters=self.model_parameters,
                                         preprocessed_name=self.preprocessed_name,
                                         network_name=self.network_name,
                                         is_inference_only=True,
                                         fmt_write=self.fmt_write,
                                         model_parameters_name=self.model_parameters_name
                                         )
        return model