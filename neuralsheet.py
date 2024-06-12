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

        retinotopic_bias = get_gaussian(self.rf_size, std_rfs).float().to(device)
        retinotopic_bias /= retinotopic_bias.view(retinotopic_bias.shape[0], -1).max(1)[0][:,None,None,None]
        self.retinotopic_bias = retinotopic_bias
        
        afferent_weights = torch.rand((sheet_size**2, 1, self.rf_size, self.rf_size), device=device)
        afferent_weights *= self.retinotopic_bias
        afferent_weights /= afferent_weights.sum([2,3], keepdim=True)
        self.afferent_weights = afferent_weights
        
        lateral_weights_exc = generate_gaussians(sheet_size, sheet_size, std_exc).to(device)
        lateral_weights_exc /= lateral_weights_exc.sum([2,3], keepdim=True)
        self.lateral_weights_exc = lateral_weights_exc
        
        untuned_inh = generate_circles(sheet_size, sheet_size, std_exc*5).to(device)
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
        self.l4_strength = 2
        self.aff_strength = 1
        
        self.iterations = 100
        self.response_tracker = torch.zeros(self.iterations, 1, sheet_size, sheet_size, device=device)

        self.avg_response = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.avg_l4_res = torch.zeros(1, 1, sheet_size, sheet_size, device=device)

        self.homeo_lr = 1e-3

        self.contrast_mean = 0
        
        self.l4_response = torch.zeros(1, 1, sheet_size, sheet_size, device=device)

        self.lri_smoother = get_gaussian(oddenise(round(std_exc*cutoff*2)), std_exc*7).to(device)
        
        self.l4_correlations = self.lateral_correlations + 0
        
        self.aff_range = 0.5
        self.res_range = 0.5
        self.l4_range = 0.5
        
        self.avg_l4_hist = torch.zeros(10) 
    
    def forward(self, input_crop, rf_grids, noise_lvl=0):

        #self.aff_b = 0.7
        self.range_norm = 0.7
        self.aff_strength = 1.8
        self.beta_response = 0.
        self.homeo_target = 0.06
        self.aff_unlearning = 0
        self.lat_unlearning = 0
        self.iterations = 100
        #self.response_tracker = torch.zeros(self.iterations, 1, self.sheet_size, self.sheet_size).cuda()
        
        self.current_response *= 0
        self.l4_response *= 0
        self.avg_response *= 0
        
        # Input crop is expected to be a 4D tensor: [batch_size, channels, N, N]
        # Process input through afferent weights
        current_input = input_crop
        self.current_input = input_crop
        self.current_tiles = extract_patches(current_input, rf_grids)
        afferent = self.current_tiles * self.afferent_weights
        afferent = afferent.sum([2,3])
        self.current_afferent = afferent.sum(1).view(self.current_response.shape)

        #untuned_inh = generate_circles(self.sheet_size, self.sheet_size, 5).to('cuda')
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
        
        #self.lri_smoother = get_gaussian(31, 0.7 * 15).to('cuda')
        
        #noise = torch.randn(self.current_response.shape, device='cuda') * noise_lvl
        noise = 0
                        
        for i in range(self.iterations):    
        
            #b = self.aff_range / (self.res_range + 1e-11)                 
            self.l4_afferent = self.current_afferent
                        
            mid_range_inhibition = self.l4_correlations * (1 - self.masks) #* self.eye
            mid_range_inhibition /= mid_range_inhibition.sum([1,2,3], keepdim=True) + 1e-11
            self.mid_range_inhibition = mid_range_inhibition
            
            #l4_lateral = F.conv2d(self.l4_response, l4_lateral_weights, padding='valid')
            #l4_lateral = l4_lateral.view(self.l4_response.shape)
                        
            #noise = torch.randn(self.l4_response.shape, device='cuda') * 0.025
            #self.l4_response = torch.relu(self.l4_afferent + l4_lateral - self.l4_thresholds) * self.l4_strength
            #self.l4_response = self.l4_response / divisive_inh * self.l4_strength
            #self.l4_response = torch.minimum(self.l4_response, torch.tensor(1))
            #self.l4_response = torch.tanh(self.l4_response)
            
            #excitation = torch.relu(self.l4_correlations - 2/self.sheet_size**2) 
            #excitation /= excitation.sum([1,2,3], keepdim=True) + 1e-11
            #self.excitation = excitation
            
            #pad = self.lri_smoother.shape[-1]//2
            #inhibition = torch.relu(self.l4_correlations - 1.5/self.sheet_size**2) 
            #inhibition = F.conv2d(excitation, self.lri_smoother, padding=pad)
            #inhibition = self.lateral_weights_inh
            #inhibition /= inhibition.sum([1,2,3], keepdim=True) + 1e-11
            #self.inhibition = inhibition
            
            #imp_start = 0
            #imp_mid = 0.1
            #imp_end = 0.5
            #idx = 0
            #beta_imp = 1.2
            
            #if i<idx:
            #    self.si = imp_start
            #elif i==idx:
            #    self.si = imp_mid
            #else:
            #    self.si = self.si*beta_imp if self.si*beta_imp<imp_end else imp_end
                
            #print(self.si)
            
            #idx = 15
            #self.si = 0.1 if i<idx else 0.5
                       
            #self.long_interactions = (self.excitation*0.02 - self.inhibition*self.si)
            interactions = self.lateral_weights_exc - mid_range_inhibition #+ self.long_interactions
            
            lateral = F.conv2d(self.current_response, interactions, padding='valid')
            lateral = lateral.view(self.current_response.shape) 
            
            net_afferent = self.l4_afferent - self.l4_thresholds  
            l4_update = net_afferent*self.aff_strength + lateral*self.l4_strength
                        
            self.l4_response = torch.relu(l4_update + noise)
            #self.l4_response = torch.minimum(torch.tensor(1.1), self.l4_response)
            self.l4_response = torch.tanh(self.l4_response) / self.range_norm
            
            max_change = (self.current_response - self.l4_response).abs().max()
            
            self.current_response = self.l4_response
            
            #divisive_inh[divisive_inh<1] = 1
            #b = self.aff_range / (self.l4_range + 1e-11)
                                                                                    
            # Initial activation
            #self.res_b = 0.5
            #self.l3_afferent = self.res_b*self.current_afferent + (1-self.res_b)*self.l4_response
            #z = torch.relu(self.l3_afferent + lateral*0.5 - self.thresholds) * self.strength
            #self.current_response = z / divisive_inh * self.strength
            #self.current_response = torch.minimum(self.current_response, torch.tensor(1))
            #self.current_response = torch.tanh(z)
            #self.current_response = self.l4_response

            self.avg_response = self.beta_response*self.avg_response + (1-self.beta_response)*self.current_response
            self.avg_l4_res = self.beta_response*self.avg_l4_res + (1-self.beta_response)*self.l4_response
            
            self.response_tracker[i] = self.current_response.clone()
            
            if max_change < 5e-3:
                break
                        
                                    
        #nonzero_act = self.current_response[self.current_response>0.1]**2
        l4_nonzero = self.l4_response[self.l4_response>0.]
        fast_lr = self.homeo_lr * 10
        if l4_nonzero.shape[0] > 1e1:
            # max is most theoretically sound because is totally independent on thresholds
            self.l4_strength -= (l4_nonzero.max() - 1) * fast_lr
            self.l4_strength = max(0, self.l4_strength)
            #aff_range = self.current_afferent.max() - self.current_afferent.min()
            #self.aff_range = self.aff_range * (1 - fast_lr) + aff_range * fast_lr
            #res_range = self.current_response.max() - self.current_response.min()
            #self.res_range = self.res_range * (1 - fast_lr) + res_range * fast_lr
            #l4_range = self.l4_response.max() - self.l4_response.min()
            #self.l4_range = self.l4_range * (1 - fast_lr) + l4_range * fast_lr
            new_hist = np.histogram(self.l4_response.cpu(), bins=10, range=(0,1))[0]
            self.avg_l4_hist = self.avg_l4_hist*(1-fast_lr) + new_hist*fast_lr
                
        #aff_nonzero = net_afferent[net_afferent>0.1]**2
        #if aff_nonzero.shape[0] > 1e1:
        #    self.aff_strength -= (aff_nonzero.mean() - 0.04) * fast_lr
            
        self.l4_mean_activations = self.l4_mean_activations * (1 - self.homeo_lr) + self.l4_response * self.homeo_lr
        self.l4_thresholds -= (self.homeo_target - self.l4_mean_activations) / self.homeo_target * self.homeo_lr
            
        #self.mean_activations = self.mean_activations * (1 - self.homeo_lr) + self.current_response * self.homeo_lr
        #self.thresholds -= (self.homeo_target - self.mean_activations) / self.homeo_target * self.homeo_lr
            
    def hebbian_step(self):

        afferent_contributions = (self.current_tiles - self.aff_unlearning) * self.retinotopic_bias 
        self.step(self.afferent_weights, afferent_contributions, self.l4_response)

        #lateral_contributions = self.avg_response - self.lat_unlearning 
        #self.step(self.lateral_correlations, lateral_contributions, self.current_response)
        
        l4_contributions = self.avg_l4_res - self.lat_unlearning 
        self.step(self.l4_correlations, l4_contributions, self.l4_response)
        

    def step(self, weights, target, response):

        delta = response.view(-1,1,1,1) * target
        weights += self.hebbian_lr * delta # add new changes
        weights *= weights > 0 # clear weak weights
        weights /= weights.sum([2,3], keepdim=True) + 1e-11 # normalise remaining weights
            
