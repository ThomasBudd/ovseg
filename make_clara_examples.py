import matplotlib.pyplot as plt
import numpy as np
from ovseg.data.Dataset import raw_Dataset
import os
from tqdm import tqdm
import cv2
from skimage.morphology import binary_dilation, binary_erosion

# pathes
plotp = os.path.join(os.environ['OV_DATA_BASE'],
                     'plots',
                     'Clara_examples')
rawp = os.path.join(os.environ['OV_DATA_BASE'],
                    'raw_data',
                    'ICON8_14_Derby_Burton')

# params for morph operations
its = 2

# colors for rois
cl_list = [1, 9]
col_list = np.array([[255, 0, 255], [0, 255, 255]]).reshape((2, 1, 1, 3))
alpha_seg_edge = 1
alpha_seg_fill = 0.5

n_ims_max = 5

im_clara = plt.imread(os.path.join(plotp, 'Clara_example.png'))[..., :3]

bbox = [80, 470, 202, 648]

im_blank = im_clara.copy()
im_blank[bbox[0]:bbox[1], bbox[2]:bbox[3], :] = 0
plt.imshow(im_blank)



# %%
plt.close()

counter = 0
ds = raw_Dataset(rawp, create_missing_labels_as_zero=True)

for data_tpl in tqdm(ds):
    
    lb = data_tpl['label']
    
    if lb.max() == 0:
        continue
    
    contains = np.where(np.sum(lb, (1,2)))[0]
    
    z_list = np.random.choice(contains,
                              replace=False,
                              size=np.min([len(contains), n_ims_max]))
    
    # window image and normalize to [0,1]
    im = (data_tpl['image'].clip(-150, 250) + 150) / 400 * 255
    # to rgb black and white image (channel last)
    im = np.stack(3*[im], -1)
    
    for z in z_list:
        
        im_z = im[z]
        
        seg = np.zeros_like(im_z)
        
        for cl, col in zip(cl_list, col_list):
            seg_z = (lb[z] == cl).astype(float)
            
            # compute edge via dial and eros
            seg_z_dial = seg_z.copy()
            for _ in range(its):
                seg_z_dial = binary_dilation(seg_z_dial)
            seg_z_eros = seg_z.copy()
            for _ in range(its):
                seg_z_eros = binary_erosion(seg_z_eros)
            
            seg[seg_z_dial,: ] = alpha_seg_edge * col
            seg[seg_z_eros,: ] = alpha_seg_fill * col

        
        seg_flat = np.max(seg, -1)
        
        im_z[seg_flat>0, :] = seg[seg_flat>0, :]
        
        
        im_z = cv2.resize(im_z,
                          dsize=(bbox[3] - bbox[2], bbox[1] - bbox[0]),
                          interpolation=cv2.INTER_NEAREST)
        
        im_blank[bbox[0]:bbox[1], bbox[2]:bbox[3], :] = im_z/255
        
        plt.imshow(im_blank)
        plt.axis('off')
        plt.savefig(os.path.join(plotp,
                                 f'clara_example_{counter}.png'),
                    bbox_inches='tight')
        plt.close()
        counter += 1