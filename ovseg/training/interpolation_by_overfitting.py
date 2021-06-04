import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ovseg.networks.validzUNet import validzUNet
from ovseg.training.loss_functions import CE_dice_pyramid_loss
from tqdm import tqdm

def interpolation_by_overfitting(im, lb, labeled_slices=None, target_DSC=99.5, max_iter=20,
                                 filters=8, base_lr=10**-2):

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if len(im.shape) == 3:
        im = im[np.newaxis]
    in_channels = im.shape[0]
    out_channels = int(lb.max()) + 1
    im = im.astype(np.float32)
    lb = np.stack([(lb == ch).astype(np.float32) for ch in range(out_channels)])

    if labeled_slices is None:
        fac = (im.shape[1] - 1) // (lb.shape[1] - 1)
        labeled_slices = list(range(0, im.shape[1], fac))

    net = validzUNet(in_ch=in_channels, out_ch=out_channels, filt=filters).to(dev)
    scaler = torch.cuda.amp.GradScaler()
    opt = torch.optim.Adam(net.parameters(), lr=base_lr)
    loss_fctn = CE_dice_pyramid_loss()
    
    
    x = torch.from_numpy(im).to(dev).unsqueeze(0)
    y = torch.from_numpy(lb).to(dev).unsqueeze(0)

    # zero padding of x makes our the cropping easier
    x = F.pad(x, (0, 0, 0, 0, net.n_z_convs, net.n_z_convs), "constant", 0)

    k = 2*net.n_z_convs + 1

    it = 0
    dsc = 0
    while dsc < target_DSC:
        with torch.cuda.amp.autocast():
            preds = []
            for i, z in tqdm(enumerate(labeled_slices)):
                xb = x[:, :, z:z+k]
                yb = y[:, :, i:i+1]
                pred = [net(xb)]
                loss = loss_fctn(pred, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 12)
                scaler.step(opt)
                scaler.update()
                net.zero_grad()
                preds.append(pred[0][0].detach().cpu().numpy())
            full_pred = np.argmax(np.concatenate(preds, 1), 0)
            dsc = 0
            for ch in range(1, out_channels):
                pred_ch = (full_pred == ch).astype(float)
                dsc += 200 * np.sum(pred_ch * lb[ch]) / np.sum(pred_ch + lb[ch])
            dsc /= out_channels-1
            it += 1
            print('{}: {:.4f}'.format(it, dsc))
            opt.param_groups[0]['lr'] = base_lr * (1 - dsc/100)
            if it > max_iter:
                break

        # training done! We should be overfitting, now evaluate
        with torch.no_grad():
            final_pred = net(x).cpu().numpy()
    return final_pred
