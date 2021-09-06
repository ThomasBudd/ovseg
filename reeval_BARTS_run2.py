from ovseg.model.SegmentationEnsemble import SegmentationEnsemble

p_names = ['pod_om_cascade_08', 'multiclass']
model_names = ['res_encoder_no_cascade', 'res_encoder_no_prg_lrn']


# %%
for p_name, model_name in zip(p_names, model_names):
    ens = SegmentationEnsemble(val_fold=list(range(5)),
                               data_name='OV04',
                               preprocessed_name=p_name,
                               model_name=model_name)
    ens.eval_raw_dataset('BARTS', save_preds=True)
    ens.clean()
