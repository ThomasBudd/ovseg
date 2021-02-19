import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from os import environ, listdir
from os.path import join, isdir
from skimage.measure import label
from ovseg.utils.interp_utils import resize_img

plt.close('all')

lrp = join(environ['OV_DATA_BASE'], 'learned_recons')
lbp = join(environ['OV_DATA_BASE'], 'raw_data', 'BARTS', 'labels')
gtp = join(environ['OV_DATA_BASE'], 'raw_data', 'BARTS', 'images')
lbp2 = join(environ['OV_DATA_BASE'], 'raw_data', 'ApolloTCGA', 'labels')
gtp2 = join(environ['OV_DATA_BASE'], 'raw_data', 'ApolloTCGA', 'images')

ps = [128, 128]


# %%
def load_im_lb(case):
    try:
        seg = nib.load(join(lbp, case)).get_fdata() > 0
        im = nib.load(join(gtp, case[:8]+'_0000.nii.gz')).get_fdata()
    except FileNotFoundError:

        try:
            seg = nib.load(join(lbp2, case)).get_fdata() > 0
            im = nib.load(join(gtp2, case[:8]+'_0000.nii.gz')).get_fdata()
        except FileNotFoundError:
            return None

    lb = label(seg)
    c_max = np.argmax([np.sum(lb == c) for c in range(1, lb.max()+1)]) + 1
    lb = (lb == c_max).astype(float)
    return lb, im


# %%
def compute_plot_coordinates(lb):
    z = np.argmax(np.sum(lb, (0, 1)))
    grid_y, grid_x = np.meshgrid(np.arange(512), np.arange(512))

    n = np.sum(lb[..., z])
    x, y = np.sum(lb[..., z] * grid_x)/n - ps[0]//2, np.sum(lb[..., z] * grid_y)/n - ps[1]//2
    x, y = x.clip(0, 512 - ps[0]), y.clip(0, 512-ps[1])
    return int(x), int(y), int(z)


def plot_recons(im, lb, recons, names, zoom=False, window=False, cbar=False):
    k = len(recons) + 1
    x, y, z = compute_plot_coordinates(lb)
    if not window:
        mn, mx = im.min(), im.max()
    else:
        mn, mx = -50, 350
        im = im.clip(mn, mx)
        recons = [recon.clip(mn, mx) for recon in recons]
    if zoom:
        xs, xe, ys, ye = x, x+ps[0], y, y+ps[1]
    else:
        xs, xe, ys, ye = 0, 512, 0, 512
    plt.subplot(2, k, 1)
    plt.imshow(im[xs:xe, ys:ye, z], cmap='gray', vmin=mn, vmax=mx)
    plt.contour(lb[xs:xe, ys:ye, z], cmap='gray')
    plt.title('Siemens')
    plt.axis('off')
    if cbar:
        plt.colorbar()
    plt.subplot(2, k, k + 1)
    plt.imshow(im[xs:xe, ys:ye, z], cmap='gray', vmin=mn, vmax=mx)
    plt.axis('off')
    if cbar:
        plt.colorbar()
    for i, (recon, name) in enumerate(zip(recons, names)):
        plt.subplot(2, k, 2+i)
        plt.imshow(recon[xs:xe, ys:ye, z], cmap='gray', vmin=mn, vmax=mx)
        plt.contour(lb[xs:xe, ys:ye, z], cmap='gray')
        plt.title(name)
        plt.axis('off')
        if cbar:
            plt.colorbar()
        plt.subplot(2, k, 2+i+k)
        plt.imshow(recon[xs:xe, ys:ye, z], cmap='gray', vmin=mn, vmax=mx)
        plt.axis('off')
        if cbar:
            plt.colorbar()

# %%

case_ids = [320, 326, 339, 379, 437, 500, 503, 511, 527]
for case_id in case_ids:
    case = 'case_{}.nii.gz'.format(case_id)
    
    # %% load everything
    out = load_im_lb(case)
    if out is None:
        continue
    else:
        lb, im = out
    a, b = 64.64038, -1.7883005
    names_HU = ['seq_HU', 'joined_HU']
    names_win = ['seq_win', 'joined_win']
    
    recon_seq_HU = nib.load(join(lrp, 'seq_HU', case)).get_fdata()
    recon_joined_HU = a * resize_img(nib.load(join(lrp, 'joined_HU', case)).get_fdata(),
                                     [512, 512, recon_seq_HU.shape[-1]], 3) + b
    recon_seq_win = nib.load(join(lrp, 'seq_win', case)).get_fdata()
    recon_joined_win = a * resize_img(nib.load(join(lrp, 'joined_win', case)).get_fdata(),
                                      [512, 512, recon_seq_HU.shape[-1]], 3) + b
    
    recons = np.stack([recon_seq_HU, recon_joined_HU, recon_seq_win, recon_joined_win])
    
    
    # %% plot HU
    plot_recons(im, lb, recons[:2], names_HU, window=True)
    plt.savefig(join(lrp,'joined_{}_HU_plot.png'.format(case[5:8])))
    plt.close()
    plot_recons(im, lb, recons[:2], names_HU, zoom=True, window=True)
    plt.savefig(join(lrp,'joined_{}_HU_plot_zoom.png'.format(case[5:8])))
    plt.close()
    # plot HU
    plot_recons(im, lb, recons[:2], names_win, window=True)
    plt.savefig(join(lrp,'joined_{}_win_plot.png'.format(case[5:8])))
    plt.close()
    plot_recons(im, lb, recons[:2], names_win, zoom=True, window=True)
    plt.savefig(join(lrp,'joined_{}_win_plot_zoom.png'.format(case[5:8])))
    plt.close()
