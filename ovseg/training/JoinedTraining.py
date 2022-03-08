from ovseg.training.TrainingBase import TrainingBase
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
from os.path import join, exists
from time import perf_counter
from torch.cuda import amp
from torch.optim import SGD, Adam

default_opt_params = {'momentum': 0.99, 'weight_decay': 3e-5, 'nesterov': True,
                      'lr': 10**-2}
default_lr_params = {'beta': 0.9, 'lr_min': 10**-6}


class JoinedTraining(TrainingBase):
    '''
    Standard network trainer e.g. for segmentation problems.
    '''

    def __init__(self, model1, model2, trn_dl,  model_path,
                 loss_weight, num_epochs=500, lr1_params=None,
                 lr2_params=None, opt1_params=None, opt2_params=None,
                 val_dl=None, opt1_name='adam', opt2_name='sgd',
                 dev='cuda', nu_ema_trn=0.99,
                 nu_ema_val=0.7, network1_name='reconstruction',
                 network2_name='segmentation', fp32=False,
                 p_plot_list=[0, 0.5, 0.8]):
        super().__init__(trn_dl, num_epochs, model_path)

        self.model1 = model1
        self.model2 = model2
        self.val_dl = val_dl
        self.loss_weight = loss_weight
        self.dev = dev
        self.nu_ema_trn = nu_ema_trn
        self.lr1_params = lr1_params if lr1_params is not None else default_lr_params
        self.lr2_params = lr2_params if lr2_params is not None else default_lr_params
        self.opt1_params = opt1_params
        self.opt2_params = opt2_params
        self.opt1_name = opt1_name
        self.opt2_name = opt2_name
        self.network1_name = network1_name
        self.network2_name = network2_name
        self.fp32 = fp32
        self.p_plot_list = p_plot_list

        self.checkpoint_attributes.extend(['nu_ema_trn', 'network1_name',
                                           'network2_name',
                                           'fp32', 'lr1_params', 'lr2_params',
                                           'opt1_params', 'opt2_params',
                                           'opt1_name', 'opt2_name',
                                           'p_plot_list'])
        # training loss
        self.trn_loss = None
        self.trn_losses = []
        self.checkpoint_attributes.append('trn_losses')
        if self.val_dl is not None:
            self.nu_ema_val = nu_ema_val
            self.val_losses = []
            self.checkpoint_attributes.extend(['val_losses', 'nu_ema_val'])

        # check if we apply augmentation on the GPU
        if self.fp32:
            self._trn_step = self._trn_step_fp32
        else:
            self._trn_step = self._trn_step_fp16
            self.scaler = amp.GradScaler()

        # get our loss function
        self.initialise_loss()

        # setup optimizer
        self.model1.network = self.model1.network.to(self.dev)
        self.model2.network = self.model2.network.to(self.dev)
        self.initialise_opt()

    def initialise_loss(self):
        print('Nothing to do. Losses are alreay initialised')

    def loss_fctn(self, out1, out2, yb1, yb2):
        loss1 = self.model1.training.loss_fctn(out1, yb1)
        loss2 = self.model2.training.loss_fctn(out2, yb2)
        combined_loss = (1-self.loss_weight) * loss1 + self.loss_weight * loss2
        return combined_loss, loss1, loss2

    def initialise_opt(self):
        if self.opt1_name is None or self.opt2_name is None:
            raise ValueError('please specify opt1/2_name')

        # create first optimizer
        print('first optimizer...')
        if self.opt1_name.lower() == 'sgd':
            print('initialise SGD')
            self.opt1 = SGD(self.model1.network.parameters(),
                            **self.opt1_params)
        elif self.opt1_name.lower() == 'adam':
            print('initialise Adam')
            self.opt1 = Adam(self.model1.network.parameters(),
                             **self.opt1_params)
        else:
            raise ValueError('Optimiser '+self.opt1_name+' was not does not '
                             'have a recognised implementation.')

        # load state dict from first optimizer
        try:
            self.opt1.load_state_dict(torch.load(
                join(self.model1.training.model_path, 'opt_parameters')))
        except FileNotFoundError:
            print('No opt1 parameters found.')

        # create second optimizer
        print('second optimizer...')
        if self.opt2_name.lower() == 'sgd':
            print('initialise SGD')
            self.opt2 = SGD(self.model2.network.parameters(),
                            **self.opt2_params)
        elif self.opt2_name.lower() == 'adam':
            print('initialise Adam')
            self.opt2 = Adam(self.model2.network.parameters(),
                             **self.opt2_params)
        else:
            raise ValueError('Optimiser '+self.opt2_name+' was not does not '
                             'have a recognised implementation.')

        # load state dict from second optimizer
        try:
            self.opt2.load_state_dict(torch.load(
                join(self.model2.training.model_path, 'opt_parameters')))
        except FileNotFoundError:
            print('No opt2 parameters found.')

        self.lr1_init = self.opt1_params['lr']
        self.lr2_init = self.opt2_params['lr']

    def update_lr(self):

        # first learning rate
        lr1 = (1-self.epochs_done/self.num_epochs)**self.lr1_params['beta'] * \
                    (self.lr1_init - self.lr1_params['lr_min']) + \
                    self.lr1_params['lr_min']
        self.opt1.param_groups[0]['lr'] = lr1
        self.print_and_log('Learning rate 1 now: {:.4e}'.format(lr1))

        # second learning rate
        lr2 = (1-self.epochs_done/self.num_epochs)**self.lr2_params['beta'] * \
            (self.lr2_init - self.lr2_params['lr_min']) + \
            self.lr2_params['lr_min']
        self.opt2.param_groups[0]['lr'] = lr2
        self.print_and_log('Learning rate 2 now: {:.4e}'.format(lr2))

    def save_checkpoint(self):
        super().save_checkpoint()
        # save network parameters
        torch.save(self.model1.network.state_dict(),
                   join(self.model_path, self.network1_name + '_weights'))
        torch.save(self.model2.network.state_dict(),
                   join(self.model_path, self.network2_name + '_weights'))
        # save optimizer state_dict
        torch.save(self.opt1.state_dict(), join(self.model_path,
                                                'opt1_parameters'))
        torch.save(self.opt2.state_dict(), join(self.model_path,
                                                'opt2_parameters'))

        # the scaler also has savable parameters
        if not self.fp32:
            torch.save(self.scaler.state_dict(), join(self.model_path,
                                                      'scaler_parameters'))

        self.print_and_log(' parameters and opt parameters saved.')

    def load_last_checkpoint(self):
        # first we try to load the training attributes
        if not super().load_last_checkpoint():
            return False

        # now let's try to load the network parameters
        net1_pp = join(self.model_path, self.network1_name+'_weights')
        if exists(net1_pp):
            self.model1.network.load_state_dict(torch.load(net1_pp))
        else:
            return False
        net2_pp = join(self.model_path, self.network2_name+'_weights')
        if exists(net2_pp):
            self.model2.network.load_state_dict(torch.load(net2_pp))
        else:
            return False

        # now the optimizer parameters, but before we should reinitialise it
        # in case the opt parameters chagened after loading
        self.initialise_opt()
        opt1_pp = join(self.model_path, 'opt1_parameters')
        if exists(opt1_pp):
            self.opt1.load_state_dict(torch.load(opt1_pp))
        else:
            return False
        opt2_pp = join(self.model_path, 'opt2_parameters')
        if exists(opt2_pp):
            self.opt2.load_state_dict(torch.load(opt2_pp))
        else:
            return False
        # load the loss function. Nothing should go wrong here
        self.initialise_loss()

        # now in the case of fp16 training we load the scaler parameters
        if not self.fp32:
            # get a new scaler and overwrite _trn_step
            # this doesn't hurt, but is usefull for the case
            # that NetworkTraining was initialised with fp32=True
            # and loaded with fp32=False
            self.scaler = amp.GradScaler()
            self._trn_step = self._trn_step_fp16
            scaler_pp = join(self.model_path, 'scaler_parameters')
            if exists(scaler_pp):
                self.scaler.load_state_dict(torch.load(scaler_pp))
            else:
                print('Warning, no state dict for fp16 scaler found. '
                      'It seems like training was continued switching from '
                      'fp32 to fp16.')
        else:
            self._trn_step = self._trn_step_fp32
        return True

    def train(self, try_continue=True):
        loaded = False
        if try_continue:
            if self.load_last_checkpoint():
                self.print_and_log('Loaded checkpoint from previous training!')
                loaded = True
        if not loaded:
            self.print_and_log('No previous checkpoint found,'
                               ' start from scrtach.')
        self.enable_autotune()
        super().train()

    def zero_grad(self):
        for param in self.model1.network.parameters():
            param.grad = None
        for param in self.model2.network.parameters():
            param.grad = None

    def _pad_or_crop_recon(self, recon, seg, coord):
        patch_size = np.array(seg.shape[-2:])
        shape = np.array(recon.shape[1:])
        if np.all(shape <= patch_size):
            # we're smaller let's pad the arrays
            pad = (0, patch_size[1] - shape[1], 0, patch_size[0] - shape[0])
            recon = F.pad(recon, pad)
        elif np.all(shape > patch_size):
            # the sample is larger, let's do a random crop!
            recon = recon[:, coord[0]:coord[0]+patch_size[0],
                          coord[1]:coord[1]+patch_size[1]]
        else:
            raise ValueError('Something weird happend when try to crop or '
                             'pad! Got shape {} and patch size '
                             '{}'.format(shape, patch_size))
        return recon

    def _eval_data_tpl(self, data_tpl):
        proj = data_tpl['projection'][0].to(self.dev)
        im_att = data_tpl['image'][0].to(self.dev)
        seg = data_tpl['label'][0].to(self.dev)
        spacing = data_tpl['spacing'][0].numpy()
        xycoords = data_tpl['xycoords'][0].numpy()
        recon = self.model1.network(proj)
        # print('recon device: '+str(recon.device))
        recon_hu = self.model1.postprocessing(recon)
        # print('recon_hu device: '+str(recon_hu.device))
        recon_prep = self.model2.preprocessing.preprocess_batch(recon_hu, spacing)
        # print('recon_prep device: '+str(recon_prep[0].device))
        # now crop recon_pred and seg
        batch = []
        for b in range(len(recon_prep)):
            # augment and crop sample by sample
            sample = self._pad_or_crop_recon(recon_prep[b], seg[b], xycoords[b])
            batch.append(torch.cat([sample, seg[b]]))
        batch = torch.stack(batch)
        # print('batch device: '+str(batch.device))
        batch = self.model2.augmentation.GPU_augmentation.augment_batch(batch)
        # print('batch_aug device: '+str(batch.device))
        recon_aug, seg_aug = batch[:, :-1], batch[:, -1:]
        pred = self.model2.network(recon_aug)
        # print('pred device: '+str(pred.device))
        return recon, pred, im_att, seg_aug

    def _trn_step_fp32(self, data_tpl):
        # classical fp32 training
        self.zero_grad()
        out1, out2, yb1, yb2 = self._eval_data_tpl(data_tpl)
        loss = self.loss_fctn(out1, out2, yb1, yb2)
        loss[0].backward()
        self.opt1.step()
        self.opt2.step()
        return loss

    def _trn_step_fp16(self, data_tpl):
        # fancy new mixed precision training of pytorch
        self.zero_grad()
        with amp.autocast():
            out1, out2, yb1, yb2 = self._eval_data_tpl(data_tpl)
            loss = self.loss_fctn(out1, out2, yb1, yb2)
        self.scaler.scale(loss[0]).backward()
        self.scaler.unscale_(self.opt1)
        self.scaler.unscale_(self.opt2)
        torch.nn.utils.clip_grad_norm_(self.model1.network.parameters(), 12)
        torch.nn.utils.clip_grad_norm_(self.model2.network.parameters(), 12)
        self.scaler.step(self.opt1)
        self.scaler.step(self.opt2)
        self.scaler.update()
        return loss

    def do_trn_step(self, data_tpl, step):

        loss = self._trn_step(data_tpl)
        # detach and to cpu
        loss = np.array([loss_item.detach().item() for loss_item in loss])
        if self.trn_loss is None:
            self.trn_loss = loss
        else:
            self.trn_loss = self.nu_ema_trn * self.trn_loss + \
                (1 - self.nu_ema_trn) * loss

    def on_epoch_start(self):
        super().on_epoch_start()
        self.model1.network.train()
        self.model2.network.train()

    def on_epoch_end(self):
        super().on_epoch_end()
        # keep the training loss from the end of the epoch
        trn_loss_ema = []
        for i in range(len(self.trn_loss)):
            if not np.isnan(self.trn_loss[i]):
                trn_loss_ema.append(self.trn_loss[i])
            else:
                self.print_and_log('Warning: computed NaN for trn loss, continuing EMA with previous '
                                   'value.')
                trn_loss_ema.append(self.trn_losses[-1][i])
                self.trn_loss[i] = self.trn_losses[-1][i]
        self.trn_losses.append(trn_loss_ema)
        self.print_and_log('Traning losses: {:.4e}, {:.4e}, '
                           '{:.4e}'.format(*self.trn_loss))

        # evaluate network on val_dl if given
        self.estimate_val_loss()

        # now make some nice plots and we're happy!
        self.plot_training_progess()
        self.update_lr()

    def estimate_val_loss(self):
        '''
        Estimates the loss on the validation set but running val_dl if one
        if given
        '''
        self.model1.network.eval()
        self.model2.network.eval()
        if self.val_dl is not None:
            val_loss = np.array([0., 0., 0.])
            st = perf_counter()
            with torch.no_grad():
                if self.fp32:
                    for data_tpl in self.val_dl:
                        out1, out2, yb1, yb2 = self._eval_data_tpl(data_tpl)
                        loss = self.loss_fctn(out1, out2, yb1, yb2)
                        loss = np.array([loss_item.detach().item()
                                         for loss_item in loss])
                        val_loss += loss
                else:
                    with torch.cuda.amp.autocast():
                        for data_tpl in self.val_dl:
                            out1, out2, yb1, yb2 = \
                                self._eval_data_tpl(data_tpl)
                            loss = self.loss_fctn(out1, out2, yb1, yb2)
                            loss = np.array([loss_item.detach().item()
                                             for loss_item in loss])
                            val_loss += loss

            val_loss = val_loss / len(self.val_dl)
            et = perf_counter()
            self.print_and_log('Validation loss: {:.4e}, {:.4e}, '
                               '{:.4e}'.format(*val_loss))
            self.print_and_log('Validation time: {:.2f} seconds'
                               .format(et-st))
            # now store the ema of the val loss
            if len(self.val_losses) == 0:
                self.val_losses.append(val_loss)
            else:
                val_loss_ema = []
                for i in range(len(val_loss)):
                    if np.isnan(self.val_losses[-1][i]) and np.isnan(val_loss[i]):
                        self.print_and_log('Warning Both previous and current val loss are NaN.')
                        val_loss_ema.append(np.nan)
                    elif np.isnan(self.val_losses[-1][i]) and not np.isnan(val_loss[i]):
                        self.print_and_log('New val loss is not NaN. Starting EMA from this value')
                        val_loss_ema.append(val_loss[i])
                    elif not np.isnan(self.val_losses[-1][i]) and np.isnan(val_loss[i]):
                        self.print_and_log('Computed NaN for val loss. Ignoring it for EMA')
                        val_loss_ema.append(self.val_losses[-1][i])
                    else:
                        val_loss_ema.append(self.nu_ema_val * self.val_losses[-1][i]
                                            + (1-self.nu_ema_val) * val_loss[i])
                self.val_losses.append(np.array(val_loss_ema))

    def plot_training_progess(self):
        for p in self.p_plot_list:
            fig = plt.figure()
            if self.plot_learning_curve(p):
                if p == 0:
                    name = 'training_progress_full.png'
                else:
                    name = 'training_progress_{:.3f}.png'.format(p)
                plt.savefig(join(self.model_path, name))
            plt.close(fig)

    def plot_learning_curve(self, p_start=0):
        '''
        plot the latest training progress, not showing the first p_start
        percent of the curve
        '''
        if p_start < 0 or p_start >= 1:
            raise ValueError('p_start must be >=0 and < 1')
        epochs = np.arange(self.epochs_done)
        n_start = np.round(self.epochs_done * p_start).astype(int)
        if n_start == self.epochs_done:
            return False
        loss_names = ['joined_loss', 'rec_loss', 'seg_loss']
        for i, loss_name in enumerate(loss_names):
            plt.subplot(1, 3, i+1)
            plt.plot(epochs[n_start:], np.array(self.trn_losses)[n_start:, i])
            plt.xlabel('epochs')
            plt.title(loss_name)
            if self.val_dl is None:
                plt.legend(['train'])
            else:
                plt.plot(epochs[n_start:], np.array(self.val_losses)[n_start:, i])
                plt.legend(['train', 'val'])
        return True

    def enable_autotune(self):
        torch.backends.cudnn.benchmark = True


# %% new variant

class JoinedTrainingV2(JoinedTraining):

    def _eval_data_tpl(self, data_tpl):
        # we change the training
        # the network here now outputs images that are already preprocessed for segmentation
        # so we don't have to do reconstruction postprocessing and segmentation preprocessing
        proj = data_tpl['projection'][0].to(self.dev)
        im_att = data_tpl['image'][0].to(self.dev)
        seg = data_tpl['label'][0].to(self.dev)
        recon = self.model1.network(proj)
        batch = torch.cat([recon, seg], 1)
        # print('batch device: '+str(batch.device))
        batch = self.model2.augmentation.GPU_augmentation.augment_batch(batch)
        # print('batch_aug device: '+str(batch.device))
        recon_aug, seg_aug = batch[:, :-1], batch[:, -1:]
        pred = self.model2.network(recon_aug)
        # print('pred device: '+str(pred.device))
        return recon, pred, im_att, seg_aug
