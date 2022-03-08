# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 10:55:39 2021

@author: Subhadip Mukherjee
"""
import torch
import numpy as np

from ovseg.preprocessing.Restauration2dSimPreprocessing import Restauration2dSimPreprocessing

class Restauration2dSimPreprocessingBowtie(Restauration2dSimPreprocessing):
    def __init__(self, photon_stat):
        Restauration2dSimPreprocessing.__init__(self,n_angles=500, source_distance=600, det_count=736, det_spacing=1.0,
                 num_photons=None, mu_water=0.0192, window=None, scaling=None,
                 fbp_filter='ramp', apply_z_resizing=True, target_z_spacing=None)
        

    def preprocess_image_bowtie(self, img, bowtie_filt, dose_level=1.0):
        '''
        Simulation of 2d sinograms and windowing/rescaling of images
        If im is the image after rescaling (and windowing) and R the Ray transform we simulate as
            proj = -1/mu x log( Poisson(n_photons x exp(-mu R(im)))/n_photons )
        mu is just a scaling constant to
        '''
        # input img must be in HU
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
            is_np = True
        elif not torch.is_tensor(img):
            raise TypeError('Input of \'preprocess_image\' must be np array '
                            'or torch tensor. Got {}'.format(type(img)))
        else:
            is_np = False

        if not len(img.shape) == 2:
            raise ValueError('Preprocessing/simulation of projection data is '
                             'only implemented for 2d images. '
                             'Got shape {}'.format(len(img.shape)))
            
            
        '''
        bowtie filter
        '''
        if isinstance(bowtie_filt, np.ndarray):
            bowtie_filt = torch.from_numpy(bowtie_filt)
            bowtie_filt = bowtie_filt.squeeze()
        else not torch.is_tensor(bowtie_filt):
            raise TypeError('The bowtie filter must be a numpy array or torch tensor. Got {}'.format(type(img)))

        
        if not len(bowtie_filt.shape) == 1:
            raise ValueError('The bowtie filter must be a 1D array'
                             'Got shape {}'.format(len(bowtie_filt.shape)))
            
        if not bowtie_filt.size()[0] == self.det_count:
            raise ValueError('The bowtie filter must have the same size as the number of detector pixels {}'
                             'Got size {}'.format(self.det_count, bowtie_filt.size()[0]))

        # we're ingoring HU < 1000
        img = img.clip(-1000)

        # rescale from HU to linear attenuation
        img_linatt = (img + 1000) / 1000 * self.mu_water
        img_linatt = img_linatt.type(torch.float).to('cuda')
        
        
        ## compute projection with bowtie filter
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        bowtie_filt = self.dose_level * self.bowtie_filt.to(dev)
        proj = self.operator.forward(img_linatt)
        proj = torch.exp(-img_linatt) 
        proj = torch.poisson(bowtie_filt.expand_as(proj)*proj)*(1/bowtie_filt.expand_as(proj))
        sinogram_noisy = -torch.log(1e-6 + proj)
        fbp_linatt = self.operator.backprojection(self.operator.filter_sinogram(sinogram_noisy))
        fbp = 1000 * (fbp_linatt - self.mu_water) / self.mu_water
        
        # now windowing and recaling
        if self.window is not None:
            img = img.clip(*self.window)
            fbp = fbp.clip(*self.window)
        if self.scaling is not None:
            img = (img - self.scaling[1]) / self.scaling[0]
            fbp = (fbp - self.scaling[1]) / self.scaling[0]

        if is_np:
            return fbp.cpu().numpy(), img.cpu().numpy()
        else:
            return fbp, img
