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
        self.rf_size = 13

        retinotopic_bias = get_gaussian(self.rf_size, std_rfs).float().to(device)
        retinotopic_bias = get_circle(self.rf_size, self.rf_size/2).float().to(device)
        retinotopic_bias /= retinotopic_bias.view(retinotopic_bias.shape[0], -1).max(1)[0][:,None,None,None]
        self.retinotopic_bias = retinotopic_bias
        
        afferent_weights = torch.rand((sheet_size**2, 1, self.rf_size, self.rf_size), device=device)
        afferent_weights *= self.retinotopic_bias
        afferent_weights /= afferent_weights.sum([2,3], keepdim=True)
        self.afferent_weights = afferent_weights
        
        lateral_weights_exc = generate_gaussians(sheet_size, sheet_size, std_exc).to(device)
        lateral_weights_exc /= lateral_weights_exc.sum([2,3], keepdim=True)
        self.lateral_weights_exc = lateral_weights_exc
        
        untuned_inh = generate_circles(sheet_size, sheet_size, max(2, std_exc*5)).to(device)
        untuned_inh /= untuned_inh.sum([2,3], keepdim=True)
        self.untuned_inh = untuned_inh 

        self.masks = \
            1 - untuned_inh/untuned_inh.view(untuned_inh.shape[0], -1).max(1)[0][:,None,None,None]

        self.eye = 1 - \
            lateral_weights_exc/lateral_weights_exc.view(lateral_weights_exc.shape[0], -1).max(1)[0][:,None,None,None]
        
        
        lateral_correlations = torch.rand((sheet_size**2, 1, sheet_size, sheet_size), device=device)
        lateral_correlations /= lateral_correlations.sum([2,3], keepdim=True)
        self.lateral_correlations = lateral_correlations
        
        # Homeostatic thresholds for each neuron
        self.l4_thresholds = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.l4_mean_activations = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
                
        self.current_response = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        
        self.l4_strength = 2
        self.aff_strength = 1
        
        self.iterations = 100
        self.response_tracker = torch.zeros(self.iterations, 1, sheet_size, sheet_size, device=device)

        self.avg_response = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.avg_l4_res = torch.zeros(1, 1, sheet_size, sheet_size, device=device)

        self.homeo_lr = 1e-3

        self.contrast_mean = 0
        
        self.l4_response = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        
        self.l4_correlations = self.lateral_correlations + 0
        
        self.aff_range = 0.5
        self.res_range = 0.5
        self.l4_range = 0.5
        
        self.avg_l4_hist = torch.zeros(10) 
        self.noise = 0
        self.std_exc = std_exc
        
        self.crop_indeces = torch.arange(sheet_size**2).to(device)
        
        N = sheet_size
        self.window = oddenise(self.std_exc*10)
        num_images = N**2
        
        batch_indices = torch.arange(num_images).view(num_images, 1, 1, 1)
        # Create a batch dimension for indices
        self.batch_indices = batch_indices.expand(num_images, 1, self.window , self.window)
        
        # Generate all possible row and column starts
        row_indices = torch.arange(0, N).repeat_interleave(N)
        col_indices = torch.arange(0, N).repeat(N)

        # Expand indices to use for gathering
        row_indices = row_indices.view(num_images, 1, 1).expand(num_images, self.window, self.window)
        col_indices = col_indices.view(num_images, 1, 1).expand(num_images, self.window, self.window)
        
        # Create range tensors for MxM crops
        range_rows = torch.arange(0, self.window).view(1, self.window, 1).expand(num_images, self.window, self.window)
        range_cols = torch.arange(0, self.window).view(1, 1, self.window).expand(num_images, self.window, self.window)

        # Add start indices and range indices
        self.final_row_indices = (row_indices + range_rows).view(num_images, 1, self.window, self.window)
        self.final_col_indices = (col_indices + range_cols).view(num_images, 1, self.window, self.window)
            
    def forward(self, input_crop, rf_grids, noise_lvl=0, adaptation=True):

        self.range_norm = 0.75
        #self.aff_strength = 1.8
        self.beta_response = 0.
        self.homeo_target = 0.04
        self.aff_unlearning = 0
        self.lat_unlearning = 0
        self.iterations = 100
        #self.response_tracker = torch.zeros(self.iterations, 1, self.sheet_size, self.sheet_size).cuda()
        
        self.current_response *= 0
        self.l4_response *= 0
        self.avg_response *= 0
        
        net_afferent = 0
        
        # Input crop is expected to be a 4D tensor: [batch_size, channels, N, N]
        # Process input through afferent weights
        current_input = input_crop
        self.current_input = input_crop
        self.current_tiles = extract_patches(current_input, rf_grids)
        afferent = self.current_tiles * self.get_aff_weights()
        afferent = afferent.sum([2,3])
        self.current_afferent = afferent.sum(1).view(self.current_response.shape)
        
        mid_range_inhibition = self.l4_correlations * (1 - self.masks)
        mid_range_inhibition /= mid_range_inhibition.sum([1,2,3], keepdim=True) + 1e-11
        self.mid_range_inhibition = mid_range_inhibition 
        interactions = self.lateral_weights_exc - mid_range_inhibition 

        # Unfold the tensor to extract sliding windows
        crops = F.pad(interactions, (self.window//2, self.window//2, self.window//2, self.window//2))
        crops = crops[self.batch_indices, :, self.final_row_indices, self.final_col_indices]
        
        #untuned_inh = generate_circles(self.sheet_size, self.sheet_size, self.std_exc*5).to('cuda')
        #untuned_inh /= untuned_inh.sum([2,3], keepdim=True)
        #self.untuned_inh = untuned_inh 
        #self.masks = \
        #    1 - untuned_inh/untuned_inh.view(untuned_inh.shape[0], -1).max(1)[0][:,None,None,None]
                                                
        for i in range(self.iterations):    
            
            
            if noise_lvl:            
                self.noise_corr = 0.8
                curr_noise = torch.randn(self.current_response.shape, device='cuda')
                self.noise = self.noise * self.noise_corr + curr_noise * (1-self.noise_corr)                 
        
            self.l4_afferent = self.current_afferent
                
            if False:
                lateral = F.conv2d(self.current_response, interactions, padding='valid')
                
            else:
                res_tiles = F.unfold(self.current_response, self.window, padding=self.window//2)[0].T
                lateral = (res_tiles * crops.view(res_tiles.shape)).sum(1)
                                                
            lateral = lateral.view(self.current_response.shape) * self.l4_strength
            
            net_afferent = (self.l4_afferent - self.l4_thresholds) * self.aff_strength
            l4_update = net_afferent + lateral
                        
            self.l4_response = torch.relu(l4_update + self.noise*noise_lvl)
            self.l4_response = torch.tanh(self.l4_response)
            
            max_change = (self.current_response - self.l4_response).abs().max()
            
            self.current_response = self.l4_response

            self.avg_response = self.beta_response*self.avg_response + (1-self.beta_response)*self.current_response
            self.avg_l4_res = self.beta_response*self.avg_l4_res + (1-self.beta_response)*self.l4_response
            
            self.response_tracker[i] = self.current_response.clone()
            
            if max_change < 3e-3:
                break
                                      
        if adaptation:
            l4_nonzero = self.l4_response[self.l4_response>0.]
            fast_lr = self.homeo_lr * 10
            if l4_nonzero.shape[0] > 1e1:
                # max is most theoretically sound because is totally independent on thresholds
                self.l4_strength -= (l4_nonzero.max() - self.range_norm) * fast_lr
                self.l4_strength = max(0, self.l4_strength)
                
                self.aff_strength -= (net_afferent.max() - 0.2) * fast_lr
                self.aff_strength = max(0, self.aff_strength)
                
                new_hist = np.histogram(self.l4_response.cpu(), bins=10, range=(0,1))[0]
                self.avg_l4_hist = self.avg_l4_hist*(1-fast_lr) + new_hist*fast_lr

            self.l4_mean_activations = self.l4_mean_activations*(1-self.homeo_lr) + self.l4_response*self.homeo_lr
            thresh_update = self.homeo_target - self.l4_mean_activations
            self.l4_thresholds -= (thresh_update/self.homeo_target)**2 * self.homeo_lr * torch.sign(thresh_update)
            
    def hebbian_step(self):

        afferent_contributions = self.current_tiles - self.aff_unlearning
        self.step(self.afferent_weights, afferent_contributions, self.l4_response)
        
        l4_contributions = self.avg_l4_res - self.lat_unlearning 
        self.step(self.l4_correlations, l4_contributions, self.l4_response)
        

    def step(self, weights, target, response):

        delta = response.view(-1,1,1,1) * target
        weights += self.hebbian_lr * delta # add new changes
        weights *= weights > 0 # clear weak weights
        weights /= weights.sum([2,3], keepdim=True) + 1e-11 # normalise remaining weights
        
    def get_aff_weights(self):
        
        aff_weights = self.afferent_weights * self.retinotopic_bias
        aff_weights /= aff_weights.sum([2,3], keepdim=True) + 1e-11
        
        return aff_weights
            
