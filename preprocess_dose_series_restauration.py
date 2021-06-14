from ovseg.preprocessing.Restauration2dSimPreprocessing import Restauration2dSimPreprocessing
import numpy as np

window = [-32, 318]
scaling = [52.286, 38.16]

dl_list = 0.5 ** np.arange(6)
ext_list = ['full', 'half', 'quater', 'eights', '16', '32'][:1]
 
for dose_level, ext in zip(dl_list, ext_list):
    preprocesseing = Restauration2dSimPreprocessing(n_angles=500,
                                                    source_distance=600,
                                                    det_count=736,
                                                    det_spacing=1.0,
                                                    mu_water=0.0192,
                                                    window=window,
                                                    scaling=scaling, fbp_filter='ramp',
                                                    apply_z_resizing=True,
                                                    target_z_spacing=5.0,
                                                    bowtie_filt=None,
                                                    dose_level=dose_level)

    fbp_folder_name = 'fbps_'+ext
    im_folder_name = 'images_restauration'

    preprocesseing.preprocess_raw_folders('OV04', 'pod_2d',
                                          fbp_folder_name=fbp_folder_name,
                                          im_folder_name=im_folder_name,
                                          save_as_fp16=True)