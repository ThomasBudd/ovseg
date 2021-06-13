from ovseg.training.NetworkTraining import NetworkTraining
import torch
from os.path import exists, join
import numpy as np

class RestaurationNetworkTraining(NetworkTraining):

    def __init__(self, compute_val_psnr_everk_k_epochs=250, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_val_psnr_everk_k_epochs = compute_val_psnr_everk_k_epochs
        if exists(join(self.model_path, 'val_psnr.npy')):
            self.val_psnr = np.load(join(self.model_path, 'val_psnr.npy')).tolist()
        else:
            self.val_psnr = []

    def initialise_loss(self):
        self.loss_fctn = torch.nn.MSELoss()

    def compute_batch_loss(self, batch):
        batch = batch.to(self.dev)
        xb, yb = batch[:, :1], batch[:, 1:]
        out = self.network(xb)
        loss = self.loss_fctn(out, yb)
        return loss

    def on_epoch_end(self):
        super().on_epoch_end()

        if self.val_dl is None:
            return

        if self.epochs_done % self.compute_val_psnr_everk_k_epochs == 0:
            self.print_and_log('Compute validatin PSNR....')
            val_ds = self.val_dl.dataset.vol_ds
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            mean_psnr = 0
            for i in range(len(val_ds)):
                data_tpl = val_ds[i]
                im = data_tpl['image']
                fbp = data_tpl['fbp']
                fbp = torch.from_numpy(fbp).to(dev)
                fbp = fbp.unsqueeze(1)

                nz = fbp.shape[0]
                bs = 1
                pred = torch.zeros((nz, 512, 512), device='cuda')
                z_list = list(range(0, nz - bs, bs)) + [nz - bs]
        
                # do the iterations
                with torch.no_grad():
                    for z in z_list:
                        batch = torch.stack([fbp[zb] for zb in range(z, z + bs)])
                        if self.fp32:
                            out = self.network(batch)
                        else:
                            with torch.cuda.amp.autocast():
                                out = self.network(batch)
                        # out back to gpu and reshape
                        out = torch.stack([out[b, 0] for b in range(bs)])
                        pred[z: z + bs] = out
                pred = pred.cpu().numpy()
                mean_psnr += 10 * np.log10(im.ptp()**2 / np.mean((im - pred) ** 2))
            mean_psnr /= len(val_ds)
            self.val_psnr.append(mean_psnr)
            self.print_and_log('Done! The mean validation PSNR is {:.4f}.'.format(mean_psnr))
