import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

plt.close('all')

fs = 14
fs_title = 18
fs_legend = 12

fac = 0.7
color_drop = [158/255, 187/255, 188/255]
color_old = [201/255, 218/255, 216/255]
color_new = np.minimum([fac * 158/255, fac*187/255, fac*188/255],1)
color_pred = [0, 0.35, 0.5]

#linewidth
lw = 2
#markersize
ms = 6
ms2 = 4
#linestyle
ls = 'dashed'

# %%
measures_kits = pickle.load(open(os.path.join(os.environ['OV_DATA_BASE'],
                                              'predictions',
                                              'kits21_trn',
                                              'disease_3_1',
                                              'all_UQ_measures.pkl'),'rb'))

measures_ov = pickle.load(open(os.path.join(os.environ['OV_DATA_BASE'],
                                            'predictions',
                                            'OV04',
                                            'pod_om_4fCV',
                                            'all_UQ_measures.pkl'),'rb'))


# %% p vs p plot

plt.subplot(1,3, 1)
plt.plot(np.arange(1,8)/7,
         measures_ov[9]['P_gt_drop'][1:8], 
         color=color_drop,
         marker='o',
         linewidth=lw,
         linestyle=ls,
         markersize=ms)
plt.plot(np.arange(1,8)/7,
         measures_ov[9]['P_gt_old'][1:8],
         color=color_old,
         marker='o',
         linewidth=lw,
         linestyle=ls,
         markersize=ms)

plt.xlabel('Estimated probability', fontsize=fs)
plt.ylabel('Prevalence of ground truth foreground', fontsize=fs)

p1 = [p1 for p2, p1 in sorted(zip(measures_ov[9]['P_gt_new'][1:8],
                                  measures_ov[9]['P'][1:8]))]
p2 = [p2 for p2, p1 in sorted(zip(measures_ov[9]['P_gt_new'][1:8],
                                  measures_ov[9]['P'][1:8]))]

plt.plot(p1,p2,
         color=color_new,
         marker='o',
         linewidth=lw,
         linestyle=ls,
         markersize=ms)
plt.plot([0, 1], [0, 1], 'k')
plt.title('Pelvic/ovarian disease', fontsize=fs_title)

plt.legend(['Dropout', 'Uncalibrated ensemble', 'Calibrated ensemble', 'Identity'],
           fontsize=fs_legend, loc='upper left')
plt.subplot(1,3, 2)
plt.plot(np.arange(1,8)/7, measures_ov[1]['P_gt_drop'][1:8],
         color=color_drop,
         marker='o',
         linewidth=lw,
         linestyle=ls,
         markersize=ms)
plt.plot(np.arange(1,8)/7, measures_ov[1]['P_gt_old'][1:8], 
         color=color_old,
         marker='o',
         linewidth=lw,
         linestyle=ls,
         markersize=ms)

p1 = [p1 for p2, p1 in sorted(zip(measures_ov[1]['P_gt_new'][1:8],
                                  measures_ov[1]['P'][1:8]))]
p2 = [p2 for p2, p1 in sorted(zip(measures_ov[1]['P_gt_new'][1:8],
                                  measures_ov[1]['P'][1:8]))]

plt.plot(p1,p2, 
         color=color_new,
         marker='o',
         linewidth=lw,
         linestyle=ls,
         markersize=ms)
plt.plot([0, 1], [0, 1], 'k')
plt.title('Omental disease', fontsize=fs_title)

plt.xlabel('Estimated probability', fontsize=fs)

plt.subplot(1,3, 3)
plt.plot(np.arange(1,8)/7, measures_kits[2]['P_gt_drop'][1:8], 
         color=color_drop,
         marker='o',
         linewidth=lw,
         linestyle=ls,
         markersize=ms)
plt.plot(np.arange(1,8)/7, measures_kits[2]['P_gt_old'][1:8], 
         color=color_old,
         marker='o',
         linewidth=lw,
         linestyle=ls,
         markersize=ms)
plt.plot(measures_kits[2]['P'][1:8],
         measures_kits[2]['P_gt_new'][1:8],
         color=color_new,
         marker='o',
         linewidth=lw,
         linestyle=ls,
         markersize=ms)
plt.plot([0, 1], [0, 1], 'k')


plt.title('Kidney tumor', fontsize=fs_title)
plt.xlabel('Estimated probability', fontsize=fs)


# %%

plt.figure()
plt.subplot(2, 3, 1)
for ext, c in zip(['drop', 'old', 'new'], [color_drop, color_old, color_new]):
    plt.plot(measures_ov[9][f'DSCs_{ext}'],
             measures_ov[9]['DSCs_gt'],
             marker='o',
             markersize=ms2,
             color=c,
             linestyle='')
plt.plot([0, 100], [0, 100], 'k')

plt.legend(['Dropout', 'Uncalibrated ensemble', 'Calibrated ensemble', 'Identity'], 
           fontsize=fs_legend, loc='upper left')
plt.title('Pelvic/ovarian disease', fontsize=fs_title)
plt.xlabel('DSC from heatmap', fontsize=fs)
plt.ylabel('DSC from ground truth', fontsize=fs)

plt.subplot(2, 3, 2)
for ext, c in zip(['drop', 'old', 'new'], [color_drop, color_old, color_new]):
    plt.plot(measures_ov[1][f'DSCs_{ext}'],
             measures_ov[1]['DSCs_gt'],
             marker='o',
             markersize=ms2,
             color=c,
             linestyle='')
plt.plot([0, 100], [0, 100], 'k')
plt.title('Omental disease', fontsize=fs_title)
plt.xlabel('DSC from heatmap', fontsize=fs)

plt.subplot(2, 3, 3)
for ext, c in zip(['drop', 'old', 'new'], [color_drop, color_old, color_new]):
    plt.plot(measures_kits[2][f'DSCs_{ext}'],
             measures_kits[2]['DSCs_gt'],
             marker='o',
             markersize=ms2,
             color=c,
             linestyle='')
plt.plot([0, 100], [0, 100], 'k')
plt.title('Kidney tumor', fontsize=fs_title)
plt.xlabel('DSC from heatmap', fontsize=fs)

plt.xlim([50, 100])
plt.ylim([50, 100])


plt.subplot(2,3,4)

p1 = [p1 for p2, p1 in sorted(zip(measures_ov[9]['P_gt_new'][1:8],
                                  measures_ov[9]['P'][1:8]))]
p2 = [p2 for p2, p1 in sorted(zip(measures_ov[9]['P_gt_new'][1:8],
                                  measures_ov[9]['P'][1:8]))]

plt.plot(p1,p2,
         color=color_new,
         marker='o',
         linewidth=lw,
         linestyle=ls,
         markersize=ms)

p1 = [p1 for p2, p1 in sorted(zip(measures_ov[9]['P_pred_new'][1:8],
                                  measures_ov[9]['P'][1:8]))]
p2 = [p2 for p2, p1 in sorted(zip(measures_ov[9]['P_pred_new'][1:8],
                                  measures_ov[9]['P'][1:8]))]

plt.plot(p1,p2,
         color=color_pred,
         marker='o',
         linewidth=lw,
         linestyle=ls,
         markersize=ms)

plt.plot([0, 1], [0, 1], 'k')
plt.xlabel('Estimated probability', fontsize=fs)
plt.ylabel('Prevalence of foreground', fontsize=fs)
plt.legend(['Ground truth', 'Prediction'], fontsize=fs_legend, loc='upper left')

plt.subplot(2,3,5)


p1 = [p1 for p2, p1 in sorted(zip(measures_ov[1]['P_gt_new'][1:8],
                                  measures_ov[1]['P'][1:8]))]
p2 = [p2 for p2, p1 in sorted(zip(measures_ov[1]['P_gt_new'][1:8],
                                  measures_ov[1]['P'][1:8]))]

plt.plot(p1,p2,
         color=color_new,
         marker='o',
         linewidth=lw,
         linestyle=ls,
         markersize=ms)

p1 = [p1 for p2, p1 in sorted(zip(measures_ov[1]['P_pred_new'][1:8],
                                  measures_ov[1]['P'][1:8]))]
p2 = [p2 for p2, p1 in sorted(zip(measures_ov[1]['P_pred_new'][1:8],
                                  measures_ov[1]['P'][1:8]))]

plt.plot(p1,p2,
         color=color_pred,
         marker='o',
         linewidth=lw,
         linestyle=ls,
         markersize=ms)
plt.plot([0, 1], [0, 1], 'k')
plt.xlabel('Estimated probability', fontsize=fs)
plt.subplot(2,3,6)

p1 = [p1 for p2, p1 in sorted(zip(measures_kits[2]['P_gt_new'][1:8],
                                  measures_kits[2]['P'][1:8]))]
p2 = [p2 for p2, p1 in sorted(zip(measures_kits[2]['P_gt_new'][1:8],
                                  measures_kits[2]['P'][1:8]))]

plt.plot(p1,p2,
         color=color_new,
         marker='o',
         linewidth=lw,
         linestyle=ls,
         markersize=ms)

p1 = [p1 for p2, p1 in sorted(zip(measures_kits[2]['P_pred_new'][1:8],
                                  measures_kits[2]['P'][1:8]))]
p2 = [p2 for p2, p1 in sorted(zip(measures_kits[2]['P_pred_new'][1:8],
                                  measures_kits[2]['P'][1:8]))]

plt.plot(p1,p2,
         color=color_pred,
         marker='o',
         linewidth=lw,
         linestyle=ls,
         markersize=ms)
plt.plot([0, 1], [0, 1], 'k')
plt.xlabel('Estimated probability', fontsize=fs)

