from ovseg.utils.io import read_dcms, save_dcmrt_from_data_tpl, read_nii

data_tpl = read_dcms('D:\\PhD\\Data\\ov_data_base\\raw_data\\BARTS_dcm\\ID_001_1')

pred,_,_ = read_nii('D:\\PhD\\Data\\ov_data_base\\predictions\\OV04\\pod_067\\larger_res_encoder'
                    '\\BARTS_ensemble_0_1_2_3_4\\case_276.nii.gz')

data_tpl['pred'] = pred

save_dcmrt_from_data_tpl(data_tpl, 'delte_me.dcm', 'pred', names=['POD'])
