import numpy as np
import pickle
from os import environ
from os.path import join
import matplotlib.pyplot as plt

p = join(environ['OV_DATA_BASE'], 'trained_models', 'OV04', 'pod_2d')

ne_list = [250, 500, 750, 1000]
names = ['2d_UNet_large_{}', '2d_UNet_small_{}']#, '2d_UNet_small_{}_v2']

for name in names:
    print(name)
    for ne in ne_list:
        modelp = join(p, name.format(ne))
        cv_res = pickle.load(open(join(modelp, 'validation_CV_results.pkl'), 'rb'))
        cv_dice = np.mean([cv_res[key]['dice_1'] for key in cv_res])
        ens_res = pickle.load(open(join(modelp, 'ensemble_0_1_2_3_4',
                                        'BARTS_results.pkl'), 'rb'))
        ens_dice = np.nanmean([ens_res[key]['dice_1'] for key in ens_res])
        mbd = 0
        for f in range(5):
            res = pickle.load(open(join(modelp, 'fold_{}'.format(f),
                                        'BARTS_results.pkl'), 'rb'))
            mbd += np.nanmean([res[key]['dice_1'] for key in res])
        mbd /= 5
        print(name.format(ne), '{:.3f}, {:.3f}, {:.3f}'.format(cv_dice, ens_dice, mbd))

# %%
doses = ['full', 'half', 'quater', 'eights', '16', '32']

for i, d in enumerate(doses):
    modelp = join(p, 'restauration_fbps_{}'.format(d), 'fold_0')
    res = pickle.load(open(join(modelp, 'validation_results.pkl'),'rb'))
    mpsnr = np.mean([res[key]['PSNR'] for key in res])
    cp = pickle.load(open(join(modelp, 'attribute_checkpoint.pkl'),'rb'))
    print(d, '{:.3f}'.format(mpsnr))
    
# %% I AM STUPID!!! I forgot the saving of the val_psnr in the first iteration of this....

def get_val_psnr_from_training_log(modelp):
    val_psnrs = []
    with open(join(modelp, 'training_log.txt')) as file:
        Lines = file.readlines()
        loi = []
        for c, line in enumerate(Lines):
            line_wd = line[21:]
            if line_wd.endswith('99\n') and line_wd.startswith('Epoch'):
                loi.append(c+9)
            if c in loi:
                psnr = line[-9:-2]
                val_psnrs.append(float(psnr))
    return val_psnrs


for i, d in enumerate(doses):
    modelp = join(p, 'restauration_fbps_{}'.format(d), 'fold_0')
    val_psnrs = get_val_psnr_from_training_log(modelp)
    plt.subplot(2, 3, i+1)
    plt.plot(range(100, 2600, 100), val_psnrs)
    plt.title(d)
    
    