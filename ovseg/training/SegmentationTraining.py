from ovseg.training.NetworkTraining import NetworkTraining
from ovseg.training.loss_functions import CE_dice_pyramid_loss, to_one_hot_encoding
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np


class SegmentationTraining(NetworkTraining):

    def __init__(self, *args,
                 prg_trn_sizes=None,
                 prg_trn_arch_params=None,
                 prg_trn_aug_params=None,
                 prg_trn_resize_on_the_fly=True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.prg_trn_sizes = prg_trn_sizes
        self.prg_trn_arch_params = prg_trn_arch_params
        self.prg_trn_aug_params = prg_trn_aug_params
        self.prg_trn_resize_on_the_fly = prg_trn_resize_on_the_fly

        # now have fun with progressive training!
        self.do_prg_trn = self.prg_trn_sizes is not None
        if self.do_prg_trn:
            if not self.prg_trn_resize_on_the_fly:
                # if we're not resizing on the fly we have to store rescaled data
                self.prg_trn_store_rescaled_data()
            self.prg_trn_n_stages = len(self.prg_trn_sizes)
            assert self.prg_trn_n_stages > 1, "please use progressive training only if you have "\
                "more then one stage."
            self.prg_trn_epochs_per_stage = self.num_epochs // self.prg_trn_n_stages
            self.prg_trn_update_parameters()
        else:
            self.prg_trn_process_batch = identity()

    def initialise_loss(self):
        self.loss_fctn = CE_dice_pyramid_loss(**self.loss_params)

    def compute_batch_loss(self, batch):

        batch = batch.cuda()
        xb, yb = batch[:, :-1], batch[:, -1:]
        xb, yb = self.prg_trn_process_batch(xb, yb)

        if self.augmentation is not None:
            batch = torch.cat([xb, yb], 1)
            with torch.no_grad():
                batch = self.augmentation(batch)
            xb, yb = batch[:, :-1], batch[:, -1:]

        yb = to_one_hot_encoding(yb, self.network.out_channels)
        out = self.network(xb)
        loss = self.loss_fctn(out, yb)
        return loss

    def prg_trn_update_parameters(self):

        if self.epochs_done == self.num_epochs:
            return

        # compute which stage we are in atm
        self.prg_trn_stage = min([self.epochs_done // self.prg_trn_epochs_per_stage,
                                  self.prg_trn_n_stages - 1])

        # this is just getting the input patch size for the current stage
        # if we use grid augmentations the out shape of the augmentation is the input size
        # for the network. Else we can take the sampled size
        if self.prg_trn_aug_params is not None:
            if 'out_shape' in self.prg_trn_aug_params:
                print_shape = self.prg_trn_aug_params['out_shape'][self.prg_trn_stage]
            else:
                print_shape = self.prg_trn_sizes[self.prg_trn_stage]
        else:
            print_shape = self.prg_trn_sizes[self.prg_trn_stage]

        self.print_and_log('\nProgressive Training: '
                           'Stage {}, size {}'.format(self.prg_trn_stage, print_shape),
                           2)

        # now set the new patch size
        if self.prg_trn_resize_on_the_fly:

            if self.prg_trn_stage < self.prg_trn_n_stages - 1:
                # the most imporant part of progressive training: we update the resizing function
                # that should make the batches smaller
                self.prg_trn_process_batch = resize(self.prg_trn_sizes[self.prg_trn_stage],
                                                    self.network.is_2d)
            else:
                # here we assume that the last stage of the progressive training has the desired
                # size i.e. the size that the augmentation/the dataloader returns
                self.prg_trn_process_batch = identity()
        else:
            # we need to change the folder the dataloader loads from
            new_folders = self.prg_trn_new_folders_list[self.prg_trn_stage]
            self.trn_dl.dataset.change_folders_and_keys(new_folders, self.prg_trn_new_keys)

        # now alter the regularization
        if self.prg_trn_arch_params is not None:
            # here we update architectural paramters, this should be dropout and stochastic depth
            # rate
            h = self.prg_trn_stage / (self.prg_trn_n_stages - 1)
            self.network.update_prg_trn(self.prg_trn_arch_params, h)

        if self.prg_trn_aug_params is not None:
            # here we update augmentation parameters. The idea is we augment more towards the
            # end of the training
            h = self.prg_trn_stage / (self.prg_trn_n_stages - 1)
            self.print_and_log('changing augmentation paramters with h={:.4f}'.format(h))
            if self.augmentation is not None:
                self.augmentation.update_prg_trn(self.prg_trn_aug_params, h, self.prg_trn_stage)
            if self.trn_dl.dataset.augmentation is not None:
                self.trn_dl.dataset.augmentation.update_prg_trn(self.prg_trn_aug_params, h,
                                                                self.prg_trn_stage)
            if self.val_dl is not None:
                if self.val_dl.dataset.augmentation is not None:
                    self.val_dl.dataset.augmentation.update_prg_trn(self.prg_trn_aug_params, h,
                                                                    self.prg_trn_stage)

    def prg_trn_store_rescaled_data(self):

        # if we don't want to resize on the fly, e.g. because the CPU loading time is the
        # bottleneck, we will resize the full volumes and save them in .npy files
        # extensions for all folders
        str_fs = '_'.join([str(p) for p in self.prg_trn_sizes[-1]])
        extensions = []
        for ps in self.prg_trn_sizes[:-1]:
            extensions.append(str_fs + '->' + '_'.join([str(p) for p in ps]))
        # the scaling factors we will use for resizing
        scales = []
        for ps in self.prg_trn_sizes[:-1]:
            scales.append(np.array(self.prg_trn_sizes[-1]) / np.array(ps))

        # let's get all the dataloaders we have for this training
        dl_list = [self.trn_dl]
        if self.val_dl is not None:
            dl_list.append(self.val_dl)

        # first let's get all the folder pathes of where we want to store the resized volumes
        ds = dl_list[0].dataset

        # maybe create the folders
        prepp = ds.vol_ds.preprocessed_path
        all_fols = []
        for fol in ds.vol_ds.folders:
            for ext in extensions:
                path_to_fol = os.path.join(prepp, fol+'_'+ext)
                all_fols.append(path_to_fol)
                if not os.path.exists(path_to_fol):
                    os.mkdir(path_to_fol)

        self.print_and_log('resize on the fly was disabled. Instead all resized volumes will be '
                           'saved at ' + prepp + ' in the following folders:')
        self.print_and_log(*all_fols)
        self.print_and_log('Checking and converting now')
        # let's look at each dl
        for dl in dl_list:

            # now we cycle through the dataset to see if there are scans we still need to resize
            for ind, scan in enumerate(ds.vol_ds.used_scans):
                convert_scan = np.any([not os.path.exists(os.path.join(fol, scan))
                                       for fol in all_fols])
                if convert_scan:
                    # at least one .npy file is missing, convert...
                    self.print_and_log('convert scan '+scan)
                    tpl = ds._get_volume_tuple(ind)
                    # let's convert only the image
                    im = tpl[0]
                    # name of the image folder, should be \'images\' most of the time
                    im_folder = ds.vol_ds.folders[ds.vol_ds.keys.index(ds.image_key)]
                    dtype = im.dtype
                    im = torch.from_numpy(im).type(torch.float).to(self.dev)
                    for scale, ext in zip(scales, extensions):
                        im_rsz = F.interpolate(im, scale_factor=scale, mode='trilinear')
                        im_rsz = im_rsz.cpu().numpy().astype(dtype)
                        np.save(os.path.join(prepp, im_folder+'_'+ext, scan), im_rsz)

                    # now the label
                    lb = tpl[-1]
                    lb_folder = ds.vol_ds.folders[ds.vol_ds.keys.index(ds.label_key)]
                    dtype = lb.dtype
                    lb = torch.from_numpy(lb).type(torch.float).to(self.dev)
                    for scale, ext in zip(scales, extensions):
                        lb_rsz = F.interpolate(lb, scale_factor=scale)
                        lb_rsz = lb_rsz.cpu().numpy().astype(dtype)
                        np.save(os.path.join(prepp, lb_folder+'_'+ext, scan), lb_rsz)

                    if len(tpl) == 3:
                        # in this case we're in the second stage and also resize the
                        # prediction from the previous stage
                        prd = tpl[1]
                        prd_folder = ds.vol_ds.folders[ds.vol_ds.keys.index(ds.pred_fps_key)]
                        dtype = prd.dtype
                        prd = torch.from_numpy(prd).type(torch.float).to(self.dev)
                        for scale, ext in zip(scales, extensions):
                            prd_rsz = F.interpolate(prd, scale_factor=scale)
                            prd_rsz = prd_rsz.cpu().numpy().astype(dtype)
                            np.save(os.path.join(prepp, prd_folder+'_'+ext, scan), prd_rsz)

        # now we need the new_keys and new_folders for each stage to update the datasets
        self.prg_trn_new_keys = [ds.image_key, ds.label_key]
        folders = [ds.vol_ds.folders[ds.vol_ds.keys.index(ds.image_key)],
                   ds.vol_ds.folders[ds.vol_ds.keys.index(ds.label_key)]]
        if ds.pred_fps_key is not None:
            self.prg_trn_new_keys.append(ds.pred_fps_key)
            folders.append(ds.vol_ds.folders[ds.vol_ds.keys.index(ds.pred_fps_key)])
        self.prg_trn_new_folders_list = []
        for ext in extensions:
            self.prg_trn_new_folders_list.append([fol+'_'+ext] for fol in folders)
        self.prg_trn_new_folders_list.append(folders)
        self.print_and_log('Done!', 1)

    def on_epoch_end(self):
        super().on_epoch_end()
        if self.do_prg_trn:
            # if we do progressive training we update the parameters....
            if self.epochs_done % self.prg_trn_epochs_per_stage == 0:
                # ... if this is the right epoch for this
                self.prg_trn_update_parameters()


class identity(nn.Identity):

    def forward(self, xb, yb):
        return xb, yb


class resize(nn.Module):

    def __init__(self, size, is_2d):
        super().__init__()

        self.size = size
        self.is_2d = is_2d
        self.mode = 'bilinear' if self.is_2d else 'trilinear'

    def forward(self, xb, yb):
        xb = F.interpolate(xb, size=self.size, mode=self.mode)
        yb = F.interpolate(yb, size=self.size)
        return xb, yb
