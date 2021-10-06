import numpy as np
from os import environ, listdir
from os.path import join
from skimage.measure import label
import nibabel as nib
from time import sleep
from tqdm import tqdm
import torch


w_list = [0.001, 0.01, 0.1]

gtp = join(environ['OV_DATA_BASE'], 'raw_data', 'OV04', 'labels')

for w in w_list:
    predp = join(environ['OV_DATA_BASE'], 'predictions', 'OV04', 'SLDS',
                 'U-Net5_'+str(w), 'cross_validation')
    n_reg = 0
    n_fg_reg = 0
    reg_sizes = []
    size = 0
    sleep(0.5)
    print()
    for case in tqdm(listdir(predp)):
        img = nib.load(join(gtp, case))
        fac = np.prod(img.header['pixdim'][1:4]) /1000
        gt = (img.get_fdata() > 0).astype(float)
        regs = (nib.load(join(predp, case)).get_fdata() == 2).astype(float)
        comps = label(regs)
        comps_gpu = torch.from_numpy(comps).cuda()
        gt_gpu = torch.from_numpy(gt).cuda()
        n_reg += comps.max()
        for c in range(1, comps.max() + 1):
            comp = (comps_gpu == c).type(torch.float)
            comp_size = comp.sum().item() * fac
            reg_sizes.append(comp_size)
            has_fg = (comp * gt_gpu).max().item()
            n_fg_reg += has_fg
            if has_fg == 0:
                size = np.max([size, comp_size])
    
    reg_sizes = np.array(reg_sizes)
    p = 100 * np.mean(reg_sizes <= size)
    
    print()
    print('w:{}, produces {} regions of which {:.2f}% showed fg'.format(w,
                                                                        n_reg,
                                                                        100 * n_fg_reg/n_reg))
    print('The largest region without fg was {:.2f}cm3, {:.2f}% of the regions can be discarded'.format(size, p))
    print()


gtp = join(environ['OV_DATA_BASE'], 'raw_data', 'BARTS', 'labels')

for w in w_list:
    predp = join(environ['OV_DATA_BASE'], 'predictions', 'OV04', 'SLDS',
                 'U-Net5_'+str(w), 'BARTS_ensemble_0_1_2_3_4')
    
    n_reg = 0
    n_fg_reg = 0
    reg_sizes = []
    size = 0
    sleep(0.5)
    print()
    for case in tqdm(listdir(predp)):
        img = nib.load(join(gtp, case))
        fac = np.prod(img.header['pixdim'][1:4]) /1000
        gt = (img.get_fdata() > 0).astype(float)
        regs = (nib.load(join(predp, case)).get_fdata() == 2).astype(float)
        comps = label(regs)
        comps_gpu = torch.from_numpy(comps).cuda()
        gt_gpu = torch.from_numpy(gt).cuda()
        n_reg += comps.max()
        for c in range(1, comps.max() + 1):
            comp = (comps_gpu == c).type(torch.float)
            comp_size = comp.sum().item() * fac
            reg_sizes.append(comp_size)
            has_fg = (comp * gt_gpu).max().item()
            n_fg_reg += has_fg
            if has_fg == 0:
                size = np.max([size, comp_size])
    
    reg_sizes = np.array(reg_sizes)
    p = 100 * np.mean(reg_sizes <= size)
    
    print()
    print('w:{}, produces {} regions of which {:.2f}% showed fg'.format(w,
                                                                        n_reg,
                                                                        100 * n_fg_reg/n_reg))
    print('The largest region without fg was {:.2f}cm3, {:.2f}% of the regions can be discarded'.format(size, p))
    print()