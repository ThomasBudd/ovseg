import nibabel as nib
import os
from tqdm import tqdm
from time import sleep


raw_names = ['kits_21', 'Lits19']


for raw_name in raw_names:
    print(raw_name)  
    
    # first copy "cross-validation" results
    predbp = os.path.join(os.environ['OV_DATA_BASE'], 'predictions',
                         f'{raw_name}_trn', 'organ', 'ground_truth')
    
    predps = [os.path.join(predbp, 'cross_validation'),
              os.path.join(predbp, f'{raw_name}_tst_ensemble_0_1_2')]
    
    rawbp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data')
    rawps = [os.path.join(rawbp, f'{raw_name}_trn'),
             os.path.join(rawbp, f'{raw_name}_tst')]

    for predp, rawp in zip(predps, rawps):
        
        print(rawp)
        sleep(0.1)    
        
        if not os.path.exists(predp):
            os.makedirs(predp)
        
        for case in tqdm(os.listdir(rawp)):
            
            img = nib.load(os.path.join(rawp, case))
            seg = (img.get_fdata() >0).astype(int)
            
            im_nii = nib.Nifti1Image(seg, img.affine, img.header)
            
            nib.save(im_nii, os.path.join(predp, case))            
