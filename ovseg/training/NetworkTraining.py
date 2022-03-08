from ovseg.training.TrainingBase import TrainingBase
import torch
import matplotlib.pyplot as plt
import numpy as np
from os.path import join, exists
from time import perf_counter
from torch.cuda import amp
from torch.optim import SGD, Adam, AdamW

default_SGD_params = {'momentum': 0.99, 'weight_decay': 3e-5, 'nesterov': True,
                      'lr': 10**-2}
default_ADAM_params = {'lr': 10**-4}
default_ADAMW_params = {'lr': 0.001, 'betas': (0.9, 0.999),
                        'eps': 1e-08, 'weight_decay': 0.01}
default_lr_params_almost_linear = {'beta': 0.9, 'lr_min': 0}
default_lr_params_lin_ascent_cos_decay = {'n_warmup_epochs': 50, 'lr_max': 0.02}
default_lr_params = {'lin_ascent_cos_decay': default_lr_params_lin_ascent_cos_decay,
                     'almost_linear': default_lr_params_almost_linear}


class NetworkTraining(TrainingBase):
    '''
    Standard network trainer e.g. for segmentation problems.
    '''

    def __init__(self, network, trn_dl,  model_path, loss_params={},
                 num_epochs=1000, opt_params=None, lr_params=None,
                 augmentation=None, val_dl=None, dev='cuda', nu_ema_trn=0.99,
                 nu_ema_val=0.7, network_name='network', fp32=False,
                 p_plot_list=[1, 0.5, 0.2], opt_name='SGD', lr_schedule='almost_linear',
                 no_bias_weight_decay=False, save_additional_weights_after_epochs=[]):
        super().__init__(trn_dl, num_epochs, model_path)

        self.network = network
        self.loss_params = loss_params
        self.val_dl = val_dl
        self.dev = dev
        self.nu_ema_trn = nu_ema_trn
        self.network_name = network_name
        self.opt_params = opt_params
        self.lr_params = lr_params
        self.augmentation = augmentation
        self.fp32 = fp32
        self.p_plot_list = p_plot_list
        self.opt_name = opt_name
        self.lr_schedule = lr_schedule
        self.no_bias_weight_decay = no_bias_weight_decay
        self.save_additional_weights_after_epochs = save_additional_weights_after_epochs
        assert self.lr_schedule in ['almost_linear', 'lin_ascent_cos_decay']
        assert isinstance(self.save_additional_weights_after_epochs, list)

        self.checkpoint_attributes.extend(['nu_ema_trn', 'network_name',
                                           'opt_params', 'fp32', 'lr_params',
                                           'p_plot_list'])
        self.print_attributes = ['model_path', 'num_epochs', 'opt_name', 'opt_params',
                                 'lr_params', 'nu_ema_trn', 'nu_ema_val', 'fp32']
        # training loss
        self.trn_loss = None
        self.trn_losses = []
        self.checkpoint_attributes.append('trn_losses')
        if self.val_dl is not None:
            self.nu_ema_val = nu_ema_val
            self.val_losses = []
            self.checkpoint_attributes.extend(['val_losses', 'nu_ema_val'])
            self.print_attributes.append('val_dl.dataset.vol_ds.scans')

        self.print_attributes.append('network')
        # check if we apply augmentation on the GPU
        if not self.fp32:
            self.scaler = amp.GradScaler()

        # check if the default parameters were changed
        if self.opt_params is None:
            self.print_and_log('No modifications from standard opt parameters'
                               ' found, load default.')

            if self.opt_name.lower() == 'sgd':
                self.opt_params = default_SGD_params
            elif self.opt_name.lower() == 'adam':
                self.opt_params = default_ADAM_params
            elif self.opt_name.lower() == 'adamw':
                self.opt_params = default_ADAMW_params
            else:
                print('Default opt params only implemented for SGD and ADAM.')
                self.opt_params = {}

            for key in self.opt_params.keys():
                self.print_and_log(key+': '+str(self.opt_params[key]))
        if self.lr_params is None:
            self.print_and_log('No modifications from standard lr parameters'
                               ' found, load default.')
            self.lr_params = default_lr_params[self.lr_schedule]
            for key in self.lr_params.keys():
                self.print_and_log(key+': '+str(self.lr_params[key]))

        # get our loss function
        self.initialise_loss()

        # setup optimizer
        self.initialise_opt()

        # now we try to load the last checkpoint
        loaded = False
        if self.load_last_checkpoint():
            self.print_and_log('Loaded checkpoint from previous training!')
            loaded = True
        if not loaded:
            self.print_and_log('No previous checkpoint found,'
                               ' start from scrtach.')

    def initialise_loss(self):
        raise NotImplementedError('initialise_loss must be implemented.')

    def compute_batch_loss(self, batch):
        raise NotImplementedError('compute_batch_loss must be implemented.')

    def initialise_opt(self):

        opt_params = self.opt_params.copy()
        if self.no_bias_weight_decay and 'weight_decay' in self.opt_params:
            # implementation of no bias weight decay
            # from https://raberrytv.wordpress.com/2017/10/29/pytorch-weight-decay-made-easy/
            l2_value = self.opt_params['weight_decay']
            del opt_params['weight_decay']      
            decay, no_decay = [], []
            for name, param in self.network.named_parameters():
                if not param.requires_grad:
                    continue # frozen weights		            
                if len(param.shape) == 1 or name.endswith(".bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)
            params = [{'params': no_decay, 'weight_decay': 0.0},
                      {'params': decay, 'weight_decay': l2_value}]
        else:
            # we don't need to do anything
            params = self.network.parameters()

        if self.opt_name is None:
            print('No specific optimiser was initialised. Taking SGD.')
            self.opt_name = 'sgd'
        if self.opt_name.lower() == 'sgd':
            print('initialise SGD')
            self.opt = SGD(params, **opt_params)
        elif self.opt_name.lower() == 'adam':
            print('initialise Adam')
            self.opt = Adam(params, **opt_params)
        elif self.opt_name.lower() == 'adamw':
            print('initialise AdamW')
            self.opt = AdamW(params, **opt_params)
        else:
            raise ValueError('Optimiser '+self.opt_name+' was not does not '
                             'have a recognised implementation.')
        self.lr_init = self.opt_params['lr']

    def update_lr(self, step=0):
        if self.lr_schedule == 'almost_linear':
            if step != -1:
                return
            lr = (1-self.epochs_done/self.num_epochs)**self.lr_params['beta'] * \
                (self.lr_init - self.lr_params['lr_min']) + \
                self.lr_params['lr_min']
            self.opt.param_groups[0]['lr'] = lr
            self.print_and_log('Learning rate now: {:.4e}'.format(lr))
        elif self.lr_schedule == 'lin_ascent_cos_decay':
            n_warm = self.lr_params['n_warmup_epochs']
            lr_max = self.lr_params['lr_max']
            if self.epochs_done < n_warm:
                lr = lr_max * (step + 1 + self.epochs_done * len(self.trn_dl)) \
                    / len(self.trn_dl) / n_warm
            else:
                if step != -1:
                    return
                lr = lr_max * np.cos(np.pi/2*(self.epochs_done - n_warm) / (self.num_epochs - n_warm))
            self.opt.param_groups[0]['lr'] = lr
            if step == -1:
                self.print_and_log('Learning rate now: {:.4e}'.format(lr))
                

    def save_checkpoint(self, path=None):
        if path is None:
            path = self.model_path
        super().save_checkpoint(path)
        # save network parameters
        torch.save(self.network.state_dict(), join(path,
                                                   self.network_name +
                                                   '_weights'))
        # save optimizer state_dict
        torch.save(self.opt.state_dict(), join(path,
                                               'opt_parameters'))

        # the scaler also has savable parameters
        if not self.fp32:
            torch.save(self.scaler.state_dict(), join(path,
                                                      'scaler_parameters'))

        self.print_and_log(self.network_name + ' parameters and opt parameters'
                           ' saved.')

        # the additioal weights after the fixed amount of epochs
        if self.epochs_done in self.save_additional_weights_after_epochs:
            
            torch.save(self.network.state_dict(), join(path,
                                                       self.network_name +
                                                       '_weights_{}'.format(self.epochs_done)))

    def load_last_checkpoint(self, path=None):
        if path is None:
            path = self.model_path
        # first we try to load the training attributes
        if not super().load_last_checkpoint(path):
            return False

        # now let's try to load the network parameters
        net_pp = join(path, self.network_name+'_weights')
        if exists(net_pp):
            self.network.load_state_dict(torch.load(net_pp))
        else:
            return False

        # now the optimizer parameters, but before we should reinitialise it
        # in case the opt parameters chagened after loading
        self.initialise_opt()
        opt_pp = join(path, 'opt_parameters')
        if exists(opt_pp):
            self.opt.load_state_dict(torch.load(opt_pp))
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
            scaler_pp = join(path, 'scaler_parameters')
            if exists(scaler_pp):
                self.scaler.load_state_dict(torch.load(scaler_pp))
            else:
                print('Warning, no state dict for fp16 scaler found. '
                      'It seems like training was continued switching from '
                      'fp32 to fp16.')
        return True

    def train(self):
        self.network = self.network.to(self.dev)
        self.enable_autotune()
        super().train()

    def zero_grad(self):
        for param in self.network.parameters():
            param.grad = None

    def do_trn_step(self, batch, step):

        self.zero_grad()
        self.update_lr(step)
        if self.fp32:
            loss = self.compute_batch_loss(batch)
            loss.backward()
            self.opt.step()
        else:
            with amp.autocast():
                loss = self.compute_batch_loss(batch)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.scaler.step(self.opt)
            self.scaler.update()

        l = loss.detach().item()
        if not np.isnan(l):
            if self.trn_loss is None:
                self.trn_loss = loss.detach().item()
            else:
                self.trn_loss = self.nu_ema_trn * self.trn_loss + \
                    (1 - self.nu_ema_trn) * loss.detach().item()

    def on_epoch_start(self):
        self.total_epoch_time = -1*perf_counter()
        super().on_epoch_start()
        self.network.train()

    def on_epoch_end(self):
        super().on_epoch_end()
        # keep the training loss from the end of the epoch
        if not np.isnan(self.trn_loss):
            self.trn_losses.append(self.trn_loss)
        else:
            self.print_and_log('Warning: computed NaN for trn loss, continuing EMA with previous '
                               'value.')
            if len(self.trn_losses) > 0:
                self.trn_losses.append(self.trn_losses[-1])
                self.trn_loss = self.trn_losses[-1]
            else:
                self.trn_losses.append(None)
                self.trn_loss = None
                
        self.print_and_log('Traning loss: {:.4e}'.format(self.trn_loss))

        # evaluate network on val_dl if given
        self.estimate_val_loss()

        # now make some nice plots and we're happy!
        self.plot_training_progess()
        self.update_lr(-1)
        self.total_epoch_time += perf_counter()
        self.print_and_log('The total epoch time was {:.2f} seconds'.format(self.total_epoch_time))

    def on_training_end(self):
        super().on_training_end()
        torch.cuda.empty_cache()

    def estimate_val_loss(self):
        '''
        Estimates the loss on the validation set but running val_dl if one
        if given
        '''
        self.network.eval()
        if self.val_dl is not None:
            val_loss = 0
            st = perf_counter()
            with torch.no_grad():
                if self.fp32:
                    for batch in self.val_dl:
                        loss = self.compute_batch_loss(batch)
                        val_loss += loss.item()
                else:
                    with torch.cuda.amp.autocast():
                        for batch in self.val_dl:
                            loss = self.compute_batch_loss(batch)
                            val_loss += loss.item()

            val_loss = val_loss / len(self.val_dl)
            et = perf_counter()
            self.print_and_log('Validation loss: {:.4e}'.format(val_loss))
            self.print_and_log('Validation time: {:.2f} seconds'
                               .format(et-st))
            # now store the ema of the val loss
            if len(self.val_losses) == 0:
                self.val_losses.append(val_loss)
            else:
                if np.isnan(self.val_losses[-1]) and np.isnan(val_loss):
                    self.print_and_log('Warning Both previous and current val loss are NaN.')
                    self.val_losses.append(np.nan)
                elif np.isnan(self.val_losses[-1]) and not np.isnan(val_loss):
                    self.print_and_log('New val loss is not NaN. Starting EMA from this value')
                    self.val_losses.append(val_loss)
                elif not np.isnan(self.val_losses[-1]) and np.isnan(val_loss):
                    self.print_and_log('Computed NaN for val loss. Ignoring it for EMA')
                    self.val_losses.append(self.val_losses[-1])
                else:
                    self.val_losses.append(self.nu_ema_val * self.val_losses[-1]
                                           + (1-self.nu_ema_val) * val_loss)


    def plot_training_progess(self):
        for p in self.p_plot_list:
            fig = plt.figure()
            if self.plot_learning_curve(p):
                if p == 1:
                    name = 'training_progress_full.png'
                else:
                    name = 'training_progress_{:.1f}%.png'.format(100*p)
                plt.savefig(join(self.model_path, name), bbox_inches='tight')
            plt.close(fig)

    def plot_learning_curve(self, p_start=0):
        '''
        plot the latest training progress, not showing the first p_start
        percent of the curve
        '''
        if p_start <= 0 or p_start > 1:
            raise ValueError('p_start must be >0 and <=1')
        epochs = np.arange(self.epochs_done)
        n_start = np.round(self.epochs_done * (1-p_start)).astype(int)
        if n_start == self.epochs_done:
            return False
        plt.plot(epochs[n_start:], self.trn_losses[n_start:])
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('trainig progress')
        if self.val_dl is None:
            plt.legend(['train'])
        else:
            plt.plot(epochs[n_start:], self.val_losses[n_start:])
            plt.legend(['train', 'val'])
        return True

    def enable_autotune(self):
        torch.backends.cudnn.benchmark = True
