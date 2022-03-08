from ovseg.model.SegmentationModelV2 import SegmentationModelV2
from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.model.ContourRefinementV3Model import ContourRefinementV3Model

data_name = 'kits21'
p_names = ['kidney_low', 'kidney_full_refine_new', 'kidney_full_refine_refine', 'kidney_tumour']
model_names = ['ps_64_bs16', 'refine_model', 'refine_model', 'first_try']
models = [SegmentationModel, ContourRefinementV3Model, ContourRefinementV3Model, SegmentationModelV2]

for p_name, model_name, model in zip(p_names, model_names, models):
    
    m = model(val_fold=0, data_name=data_name, preprocessed_name=p_name,
              model_name=model_name)
    
    m.eval_raw_dataset()


