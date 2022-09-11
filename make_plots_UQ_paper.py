import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

plt.close('all')

fs = 14
fs_title = 18
fs_legend = 12
fs_label = 10
lp = 0

fac = 0.7
color_drop = [220/255, 187/255, 188/255]
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
                                              'all_UQ_measures_v3.pkl'),'rb'))

measures_ov = pickle.load(open(os.path.join(os.environ['OV_DATA_BASE'],
                                            'predictions',
                                            'OV04',
                                            'pod_om_4fCV',
                                            'all_UQ_measures_v3.pkl'),'rb')) 

# %% p vs p plot

plt.subplot(1,3, 1)
plt.plot(np.arange(1,8)/7,
         measures_ov[9]['P_gt_new'][:-1], 
         color=color_new,
         marker='o',
         linewidth=lw,
         linestyle=ls,
         markersize=ms)

plt.plot(np.arange(1,8)/7,
         measures_ov[9]['P_gt_old'][:-1],
         color=color_old,
         marker='o',
         linewidth=lw,
         linestyle=ls,
         markersize=ms)
plt.plot(np.arange(1,8)/7,
         measures_ov[9]['P_gt_drop'][:-1], 
         color=color_drop,
         marker='o',
         linewidth=lw,
         linestyle=ls,
         markersize=ms)
plt.xlabel('Estimated probability', fontsize=fs)
plt.ylabel('Prevalence of ground truth foreground', fontsize=fs)


plt.plot([0, 1], [0, 1], 'k')
plt.title('Pelvic/ovarian disease', fontsize=fs_title)

plt.legend(['Calibrated ensemble', 'Uncalibrated ensemble', 'Dropout', 'Identity'],
           fontsize=fs_legend, loc='upper left')
plt.subplot(1,3, 2)
plt.plot(np.arange(1,8)/7,
         measures_ov[1]['P_gt_new'][:-1], 
         color=color_new,
         marker='o',
         linewidth=lw,
         linestyle=ls,
         markersize=ms)
plt.plot(np.arange(1,8)/7, measures_ov[1]['P_gt_old'][:-1], 
         color=color_old,
         marker='o',
         linewidth=lw,
         linestyle=ls,
         markersize=ms)
plt.plot(np.arange(1,8)/7,
         measures_ov[1]['P_gt_drop'][:-1],
         color=color_drop,
         marker='o',
         linewidth=lw,
         linestyle=ls,
         markersize=ms)

plt.plot([0, 1], [0, 1], 'k')
plt.title('Omental disease', fontsize=fs_title)

plt.xlabel('Estimated probability', fontsize=fs)
plt.text(0, 1, 'Underestimation', fontsize=fs)
plt.text(0.6, 0, 'Overestimation', fontsize=fs)


plt.subplot(1,3, 3)
plt.plot(np.arange(1,8)/7,
         measures_kits[2]['P_gt_new'][:-1],
         color=color_new,
         marker='o',
         linewidth=lw,
         linestyle=ls,
         markersize=ms)
plt.plot(np.arange(1,8)/7, measures_kits[2]['P_gt_old'][:-1], 
         color=color_old,
         marker='o',
         linewidth=lw,
         linestyle=ls,
         markersize=ms)
plt.plot(np.arange(1,8)/7, 
         measures_kits[2]['P_gt_drop'][:-1], 
         color=color_drop,
         marker='o',
         linewidth=lw,
         linestyle=ls,
         markersize=ms)
plt.plot([0, 1], [0, 1], 'k')


plt.title('Kidney tumor', fontsize=fs_title)
plt.xlabel('Estimated probability', fontsize=fs)


# %%

def plot_regression(x, y, color, alpha):
    x, y = np.array(x)[:, 0], np.array(y)
    x_bar, y_bar = np.mean(x), np.mean(y)
    a = np.sum((x-x_bar) * (y-y_bar)) / np.sum((x-x_bar)**2)
    b = y_bar - a*x_bar
    
    # plt.plot([0, 100], [b, 100*a+b], color=color, alpha=alpha)
    
    print(np.corrcoef(x, y)[0,1])
# %%
 
plt.figure()
plt.subplot(3, 3, 1)
print('Pelvic/ovarian')
for ext, c in zip(['new', 'old', 'drop'], [color_new, color_old, color_drop]):
    plt.plot(measures_ov[9][f'DSCs_{ext}'],
             measures_ov[9]['DSCs_gt'],
             marker='o',
             markersize=ms2,
             color=c,
             linestyle='')

for ext, c in zip(['new', 'old', 'drop'], [color_new, color_old, color_drop]):
    print(ext)
    plot_regression(measures_ov[9][f'DSCs_{ext}'], measures_ov[9]['DSCs_gt'],
                    color=c, alpha=0.5)
    
plt.plot([0, 100], [0, 100], 'k')

plt.legend(['Calibrated ensemble', 'Uncalibrated ensemble', 'Dropout', 'Identity'], 
           fontsize=fs_legend, loc='upper left')
plt.title('Pelvic/ovarian disease', fontsize=fs_title)
plt.xlabel('DSC from heatmap', fontsize=fs_label, labelpad=lp)
plt.ylabel('DSC from ground truth', fontsize=fs_label)
plt.xlim([0, 100])
plt.ylim([0, 100])

plt.subplot(3, 3, 2)
print('Omental')
for ext, c in zip(['new', 'old', 'drop'], [color_new, color_old, color_drop]):
    print(ext)
    plt.plot(measures_ov[1][f'DSCs_{ext}'],
             measures_ov[1]['DSCs_gt'],
             marker='o',
             markersize=ms2,
             color=c,
             linestyle='')
    plot_regression(measures_ov[1][f'DSCs_{ext}'], measures_ov[1]['DSCs_gt'],
                    color=c, alpha=0.5)
plt.plot([0, 100], [0, 100], 'k')
plt.title('Omental disease', fontsize=fs_title)
plt.xlabel('DSC from heatmap', fontsize=fs_label, labelpad=lp)
plt.xlim([0, 100])
plt.ylim([0, 100])

plt.subplot(3, 3, 3)
print('Kidney')
for ext, c in zip(['new', 'old', 'drop'], [color_new, color_old, color_drop]):
    print(ext)
    plt.plot(measures_kits[2][f'DSCs_{ext}'],
             measures_kits[2]['DSCs_gt'],
             marker='o',
             markersize=ms2,
             color=c,
             linestyle='')
    plot_regression(measures_kits[2][f'DSCs_{ext}'], measures_kits[2]['DSCs_gt'],
                    color=c, alpha=0.5)
plt.plot([0, 100], [0, 100], 'k')
plt.title('Kidney tumor', fontsize=fs_title)
plt.xlabel('DSC from heatmap', fontsize=fs_label, labelpad=lp)

plt.xlim([50, 100])
plt.ylim([50, 100])


plt.subplot(3,3,4)

p1 = [p1 for p2, p1 in sorted(zip(measures_ov[9]['P_gt_new'][1:8],
                                  measures_ov[9]['P'][1:8]))]
p2 = [p2 for p2, p1 in sorted(zip(measures_ov[9]['P_gt_new'][1:8],
                                  measures_ov[9]['P'][1:8]))]

plt.plot(np.arange(1,8)/7,
         measures_ov[9]['P_gt_new'][:-1],
         color=color_new,
         marker='o',
         linewidth=lw,
         linestyle=ls,
         markersize=ms)

plt.plot(np.arange(1,8)/7, measures_ov[9]['P_pred_new'][:-1],
         color=color_pred,
         marker='o',
         linewidth=lw,
         linestyle=ls,
         markersize=ms)

plt.plot([0, 1], [0, 1], 'k')
plt.xlabel('Estimated probability (calibrated ensemble)', fontsize=fs_label, labelpad=lp)
plt.ylabel('Prevalence of foreground\n (predicted and ground truth)', fontsize=fs_label)
plt.legend(['Ground truth', 'Prediction'], fontsize=fs_legend, loc='upper left')

plt.subplot(3,3,5)

plt.plot(np.arange(1,8)/7, measures_ov[1]['P_gt_new'][:-1],
         color=color_new,
         marker='o',
         linewidth=lw,
         linestyle=ls,
         markersize=ms)

plt.plot(np.arange(1,8)/7, measures_ov[1]['P_pred_new'][:-1],
         color=color_pred,
         marker='o',
         linewidth=lw,
         linestyle=ls,
         markersize=ms)
plt.plot([0, 1], [0, 1], 'k')
plt.xlabel('Estimated probability (calibrated ensemble)', fontsize=fs_label, labelpad=lp)
plt.text(0, 0.9, 'Oversegmentation', fontsize=fs_label)
plt.text(0.65, 0.05, 'Undersegmentation', fontsize=fs_label)
plt.subplot(3,3,6)

plt.plot(np.arange(1,8)/7, measures_kits[2]['P_gt_new'][:-1],
         color=color_new,
         marker='o',
         linewidth=lw,
         linestyle=ls,
         markersize=ms)


plt.plot(np.arange(1,8)/7, measures_kits[2]['P_pred_new'][:-1],
         color=color_pred,
         marker='o',
         linewidth=lw,
         linestyle=ls,
         markersize=ms)
plt.plot([0, 1], [0, 1], 'k')
plt.xlabel('Estimated probability (calibrated ensemble)', fontsize=fs_label, labelpad=lp)

# %%
predp = os.path.join(os.environ['OV_DATA_BASE'],
                     'predictions',
                     'OV04',
                     'pod_om_4fCV')
data= pickle.load(open(os.path.join(predp, 'violin_data.pkl'), 'rb'))

for c, cl in enumerate([9, 1]):
    i_list = ['Trainee FP', 'Trainee FN']#, 'Trainee TP', 'Trainee TN']
    
    violin_data = [[] for _ in range(len(i_list))]
    n = 10 ** 5
    for j, i in enumerate(i_list):
        
        n0 = np.sum([data[cl][i][key] for key in data[cl][i]])
        
        for key in data[cl][i]:
            if i == 'Trainee TN' and key == 0.0:
                continue
            
            violin_data[j].extend(int(data[cl][i][key] / 100 + 0.5) * [key])
    
    plt.subplot(3,3,c+7)
    plt.boxplot(violin_data)
    # plt.violinplot(violin_data)
    plt.xticks([1,2], ['Trainee FP', 'Trainee FN'], fontsize=fs_label)#, 'Trainee TP', 'Trainee TN'])
    # plt.title('Pelvic/ovarian disease' if cl == 9 else 'Omental disease')
    if c == 0:
        plt.ylabel('Probabilties from heatmap\n (calibrated ensemble)', fontsize=fs_label)
