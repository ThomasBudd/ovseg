from ovseg.model.SegmentationModel import SegmentationModel
from ovseg.utils.io import read_data_tpl_from_nii, read_dcms
import os

segmentation = SegmentationModel(0, 'OV04', 'segmentation_on_Siemens_recons', is_inference_only=True)

# %%
# nii_folder = 'D:\\PhD\\Data\\ov_data_base\\raw_data\\ApolloTCGA'
# nii_data_tpl = read_data_tpl_from_nii(nii_folder, 387)
# segmentation(nii_data_tpl)

# %%
dcm_folder = 'D:\\PhD\\Data\\NEW_Barts_segmentations_VB_RW\\ID_001_1'
dcm_data_tpl = read_dcms(dcm_folder)
segmentation(dcm_data_tpl)
segmentation.save_prediction(dcm_data_tpl, 'test_dataset', 'delete_me')
