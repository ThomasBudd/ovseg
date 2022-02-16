from ovseg.model.SegmentationEnsemble import SegmentationEnsemble


model_names = ['larger_res_encoder_no_prg_lrn']
p_names = ['om_067']

for p_name, model_name in zip(p_names, model_names):
    
    ens = SegmentationEnsemble(val_fold=list(range(5)),
                               data_name='OV04',
                               model_name=model_name, 
                               preprocessed_name=p_name)
    
    ens.eval_raw_dataset('BARTS', save_preds=True, force_evaluation=True)

