import os
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
import torch.nn.functional as F
from PIL import Image
import random

from wiring_efficiency_utils import *

class NeuralSheet(nn.Module):
    def __init__(
        self, 
        sheet_size, 
        input_size, 
        std_exc, 
        std_rfs,
        cutoff=5,
        device='cuda'
    ):
        super().__init__()
        self.sheet_size = sheet_size  # Size of the sheet
        self.input_size = input_size  # Size of the input crop
        self.device = device
        
        # Afferent (receptive field) weights for each neuron in the sheet
        self.rf_size = round(std_rfs * cutoff)
        self.rf_size = self.rf_size if self.rf_size%2==1 else self.rf_size+1

        retinotopic_bias = generate_gaussians(1, self.rf_size, std_rfs).to(device)
        retinotopic_bias /= retinotopic_bias.view(retinotopic_bias.shape[0], -1).max(1)[0][:,None,None,None]
        self.retinotopic_bias = retinotopic_bias
        
        afferent_weights = torch.rand((sheet_size**2, 1, self.rf_size, self.rf_size), device=device)
        afferent_weights *= self.retinotopic_bias
        afferent_weights /= afferent_weights.sum([2,3], keepdim=True)
        self.afferent_weights = afferent_weights
        
        lateral_weights_exc = generate_gaussians(sheet_size, sheet_size, std_exc).to(device)
        lateral_weights_exc /= lateral_weights_exc.sum([2,3], keepdim=True)
        self.lateral_weights_exc = lateral_weights_exc
        
        untuned_inh = generate_gaussians(sheet_size, sheet_size, max(std_exc*3,  0.7)).to(device)
        untuned_inh /= untuned_inh.sum([2,3], keepdim=True)
        self.untuned_inh = untuned_inh 

        self.masks = \
            1 - untuned_inh/untuned_inh.view(untuned_inh.shape[0], -1).max(1)[0][:,None,None,None]

        #self.eye = 1 - generate_gaussians(self.sheet_size, self.sheet_size, 0.01).to(device)
        self.eye = 1 - \
            lateral_weights_exc/lateral_weights_exc.view(lateral_weights_exc.shape[0], -1).max(1)[0][:,None,None,None]

        log_std = 5
        log_size = oddenise(cutoff*log_std)
        self.contrast_log = - get_log(log_size, log_std).to(device)
        
        
        lateral_correlations = torch.rand((sheet_size**2, 1, sheet_size, sheet_size), device=device)
        lateral_correlations /= lateral_correlations.sum([2,3], keepdim=True)
        self.lateral_correlations = lateral_correlations
        
        # Homeostatic thresholds for each neuron
        self.thresholds = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.l4_thresholds = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        #self.off_strengths = torch.ones(1, 1, sheet_size, sheet_size, device=device) 
        self.off_strengths = 1.

        
        self.mean_activations = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.l4_mean_activations = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.mean_pos_afferent = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        
        self.current_response = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        
        self.strength = 1
        self.l4_strength = 1
        self.aff_strength = 1
        
        self.iterations = 50
        self.response_tracker = torch.zeros(self.iterations, 1, sheet_size, sheet_size, device=device)

        self.avg_response = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.avg_l4_res = torch.zeros(1, 1, sheet_size, sheet_size, device=device)

        self.homeo_lr = 1e-3

        self.contrast_mean = 0
        
        self.l4_response = torch.zeros(1, 1, sheet_size, sheet_size, device=device)

        self.lri_smoother = get_gaussian(oddenise(round(std_exc*cutoff*2)), std_exc*2).to(device)
        
        self.l4_correlations = self.lateral_correlations + 0
        
        self.aff_range = 0.5
        self.res_range = 0.5
        self.l4_range = 0.5
        
        self.avg_l4_hist = torch.zeros(10) 
    
    def forward(self, input_crop, rf_grids):

        self.aff_b = 0.9
        self.beta_response = 0.
        self.homeo_target = 0.075
        self.aff_unlearning = 0
        self.lat_unlearning = self.aff_unlearning
        self.iterations = 30
        #self.response_tracker = torch.zeros(self.iterations, 1, self.sheet_size, self.sheet_size).cuda()
        
        self.current_input = input_crop
        self.current_response *= 0
        self.l4_response *= 0
        self.avg_response *= 0
        # Input crop is expected to be a 4D tensor: [batch_size, channels, N, N]
        # Process input through afferent weights
        self.current_tiles = extract_patches(input_crop, rf_grids)

        afferent = self.current_tiles * self.afferent_weights
        afferent = afferent.sum([2,3])
        self.current_afferent = afferent.sum(1).view(self.current_response.shape)

        #untuned_inh = generate_gaussians(self.sheet_size, self.sheet_size, max(5*0.1, 1)).to('cuda')
        #untuned_inh /= untuned_inh.sum([2,3], keepdim=True)
        #self.untuned_inh = untuned_inh
        #self.masks = \
        #    1 - untuned_inh/untuned_inh.view(untuned_inh.shape[0], -1).max(1)[0][:,None,None,None]

        #contrast = F.conv2d(input_crop, self.contrast_log, padding='same')
        #contrast = F.interpolate(contrast, self.sheet_size, mode='bilinear')

        #self.contrast_mean = self.contrast_mean*(1-self.homeo_lr) + contrast.mean()*self.homeo_lr
        #contrast = contrast - self.contrast_mean

        #self.contrast_fact = torch.sigmoid(contrast * 20)
        #elf.contrast_fact = contrast / 3
        #print(self.contrast_fact.mean(), self.contrast_fact.min(), self.contrast_fact.max(), 'a')
        #plt.imshow(torch.relu(contrast[0,0]).cpu())
        #plt.show()
        
        #self.lri_smoother = get_gaussian(oddenise(round(0.7*5*2)), 0.7*2).to('cuda')
                        
        for i in range(self.iterations):
            
            imp_start = 1
            imp_mid = 1.
            imp_end = 1.
            
            idx_start = 10
            imp_dec = 0.95
            
            if i<idx_start:
                self.si = imp_start
                
            elif i==idx_start:
                self.si = imp_mid
                
            else:
                self.si = self.si*imp_dec if self.si*imp_dec>imp_end else imp_end            
            
            b = self.aff_range / (self.res_range + 1e-11)                 
            self.l4_afferent = self.current_afferent * self.aff_b + self.current_response * b * (1-self.aff_b)
                        
            mid_range_inhibition = self.l4_correlations * (1 - self.masks) * self.eye
            mid_range_inhibition /= mid_range_inhibition.sum([1,2,3], keepdim=True) + 1e-11
            self.mid_range_inhibition = mid_range_inhibition
            
            l4_lateral_weights = self.lateral_weights_exc - mid_range_inhibition
            l4_lateral = F.conv2d(self.l4_response, l4_lateral_weights, padding='valid')
            l4_lateral = l4_lateral.view(self.l4_response.shape)
                        
            self.l4_response = torch.relu(self.l4_afferent + l4_lateral - self.l4_thresholds) * self.l4_strength
            #self.l4_response = self.l4_response / divisive_inh * self.l4_strength
            #self.l4_response = torch.minimum(self.l4_response, torch.tensor(1))
            self.l4_response = torch.tanh(self.l4_response)
            
            excitation = torch.relu(self.lateral_correlations - 1.5/self.sheet_size**2) 
            excitation /= excitation.sum([1,2,3], keepdim=True) + 1e-11
            self.excitation = excitation
            
            pad = self.lri_smoother.shape[-1]//2
            inhibition = torch.relu(self.lateral_correlations - 1.5/self.sheet_size**2) 
            inhibition /= inhibition.sum([1,2,3], keepdim=True) + 1e-11
            self.inhibition = inhibition
            
            long_interaction = self.excitation - torch.exp(self.inhibition) + 1
            
            lateral = F.conv2d(self.current_response, long_interaction, padding='valid')
            lateral = lateral.view(self.current_response.shape) 
            
            divisive_inh = F.conv2d(self.current_response, self.inhibition, padding='valid')
            divisive_inh = divisive_inh.view(self.current_response.shape) 
            divisive_inh = divisive_inh*3 + 1
            
            #divisive_inh[divisive_inh<1] = 1
            #b = self.aff_range / (self.l4_range + 1e-11)
                                                                                    
            # Initial activation
            z = torch.relu(self.l4_response + lateral - self.thresholds) * self.strength
            #self.current_response = z / divisive_inh * self.strength
            #self.current_response = torch.minimum(self.current_response, torch.tensor(1))
            self.current_response = torch.tanh(self.current_response)
            #self.current_response = self.l4_response

            self.avg_response = self.beta_response*self.avg_response + (1-self.beta_response)*self.current_response
            self.avg_l4_res = self.beta_response*self.avg_l4_res + (1-self.beta_response)*self.l4_response
            
            self.response_tracker[i] = self.current_response.clone()
                        
                                    
        nonzero_act = self.current_response[self.current_response>0.1]
        l4_nonzero = self.l4_response[self.l4_response>0.1]
        fast_lr = self.homeo_lr * 10
        if nonzero_act.shape[0] > 1e1 and l4_nonzero.shape[0] > 1e1:
            self.strength -= (nonzero_act.mean() - 0.5) * fast_lr
            self.l4_strength -= (l4_nonzero.mean() - 0.3) * fast_lr
            aff_range = self.current_afferent.max() - self.current_afferent.min()
            self.aff_range = self.aff_range * (1 - fast_lr) + aff_range * fast_lr
            res_range = self.current_response.max() - self.current_response.min()
            self.res_range = self.res_range * (1 - fast_lr) + res_range * fast_lr
            l4_range = self.l4_response.max() - self.l4_response.min()
            self.l4_range = self.l4_range * (1 - fast_lr) + l4_range * fast_lr
            
            new_hist = np.histogram(self.l4_response.cpu(), bins=10, range=(0,1))[0]
            self.avg_l4_hist = self.avg_l4_hist*(1-fast_lr) + new_hist*fast_lr
            
        self.l4_mean_activations = self.l4_mean_activations * (1 - self.homeo_lr) + self.l4_response * self.homeo_lr
        self.l4_thresholds -= (self.homeo_target - self.l4_mean_activations) / self.homeo_target * self.homeo_lr
            
        self.mean_activations = self.mean_activations * (1 - self.homeo_lr) + self.current_response * self.homeo_lr
        self.thresholds -= (self.homeo_target - self.mean_activations) / self.homeo_target * self.homeo_lr
            
    def hebbian_step(self):

        afferent_contributions = (self.current_tiles - self.aff_unlearning) * self.retinotopic_bias 
        self.step(self.afferent_weights, afferent_contributions, self.l4_response)

        lateral_contributions = self.avg_response - self.lat_unlearning 
        self.step(self.lateral_correlations, lateral_contributions, self.current_response)
        
        l4_contributions = self.avg_l4_res - self.lat_unlearning 
        self.step(self.l4_correlations, l4_contributions, self.l4_response)
        

    def step(self, weights, target, response):

        delta = response.view(-1,1,1,1) * target
        weights += self.hebbian_lr * delta # add new changes
        weights *= weights > 0 # clear weak weights
        weights /= weights.sum([2,3], keepdim=True) + 1e-11 # normalise remaining weights
            
