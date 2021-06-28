from ovseg.preprocessing.ClassEnsemblePreprocessing import ClassEnsemblePreprocessing
from ovseg.model.SegmentationEnsemble import SegmentationEnsemble

target_spacing = [5.0, 1.0, 1.0]
lb_classes = [1, 9]

prev_stages = [{'data_name': 'OV04',
                'preprocessed_name': 'pod_067',
                'model_name': 'larger_res_encoder'},
               {'data_name': 'OV04',
                'preprocessed_name': 'om_08',
                'model_name': 'res_encoder_no_prg_lrn'}]

for stage in prev_stages:
    ens = SegmentationEnsemble(val_fold=list(range(5)), **stage)
    ens.fill_cross_validation()
    ens.clean()
    
prep = ClassEnsemblePreprocessing(prev_stages=prev_stages,
                                  apply_resizing=True,
                                  apply_pooling=False,
                                  apply_windowing=True,
                                  scaling=[50.89403, 41.536415],
                                  window=[-36, 314],
                                  lb_classes=lb_classes,
                                  target_spacing=target_spacing)
prep.plan_preprocessing_raw_data('OV04')
prep.preprocess_raw_data('OV04', 'pod_om_10')


