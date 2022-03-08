from ovseg.utils.io import read_dcms
import pydicom

dcmp = 'D:\\PhD\\Data\\Apollo_Hilal\\AP-P5L4'
data_tpl = read_dcms(dcmp)
dcmrtp = 'D:\\PhD\\Data\\ov_data_base\\raw_data\\dcm_rt_test\\dcms\\rtstruct_test.dcm'
rt = pydicom.dcmread(dcmrtp)

# %%
for seq in rt.ROIContourSequence:
    for cs in seq.ContourSequence:
        z = int(cs.ContourImageSequence[0].ReferencedSOPInstanceUID.split('.')[-1])
        cs.ContourImageSequence[0].ReferencedSOPInstanceUID = pydicom.uid.UID(data_tpl['SOP_ids'][-z])

for ssrs in rt.StructureSetROISequence:
    ssrs.ReferencedFrameOfReferenceUID = pydicom.uid.UID('1.3.6.1.4.1.14519.5.2.1.5472.5801.300043911151544594282566061965')

print(seq)
pydicom.dcmwrite('restruct_test_3.dcm', rt)

# %%
real_rt = pydicom.dcmread('D:\\PhD\\Data\\ov_data_base\\raw_data\\dcm_rt_test\\dcms\\AP-P5L4_1975_01_20_2-ABDOMENPELVIS-20904.dcm')
cs_fake = rt.ROIContourSequence[0].ContourSequence[0]
cs_real = real_rt.ROIContourSequence[0].ContourSequence[0]


