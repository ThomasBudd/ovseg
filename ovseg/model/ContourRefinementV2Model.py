from ovseg.preprocessing.ContourRefinementV2Preprocessing import ContourRefinementV2Preprocessing
from ovseg.model.RegionexpertModel import RegionexpertModel

class ContourRefinementV2Model(RegionexpertModel):
    
     def _create_preprocessing_object(self):
        
        self.preprocessing = ContourRefinementV2Preprocessing(**self.model_parameters['preprocessing'])
