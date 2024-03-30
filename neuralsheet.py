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
        std_e, 
        device='cuda'
    ):
        super().__init__()
        self.sheet_size = sheet_size  # Size of the sheet
        self.input_size = input_size  # Size of the input crop
        self.device = device
        
        retinotopic_bias = generate_gaussians(sheet_size, input_size, 10).to(device)
        retinotopic_bias /= retinotopic_bias.max()
        self.retinotopic_bias = retinotopic_bias
        
        lateral_bias = generate_gaussians(sheet_size, sheet_size, self.sheet_size/2).to(device)
        lateral_bias /= lateral_bias.max()
        self.lateral_bias = lateral_bias
        
        # Afferent (receptive field) weights for each neuron in the sheet
        afferent_weights = torch.rand((sheet_size**2, 1, input_size, input_size), device=device)
        afferent_weights += retinotopic_bias
        afferent_weights /= afferent_weights.sum([2,3], keepdim=True)
        self.afferent_weights = afferent_weights
        
        lateral_weights_exc = generate_gaussians(sheet_size, sheet_size, std_e).to(device)
        lateral_weights_exc /= lateral_weights_exc.sum([2,3], keepdim=True)
        self.lateral_weights_exc = lateral_weights_exc
        
        untuned_inh = generate_gaussians(sheet_size, sheet_size, 2*std_e+0.5).to(device)
        untuned_inh /= untuned_inh.sum([2,3], keepdim=True)
        self.untuned_inh = untuned_inh
        
        #lateral_bias = generate_gaussians(sheet_size, sheet_size, std_e*2+1).to(device)
        
        #lateral_correlations = generate_gaussians(sheet_size, sheet_size, 2*std_e+1).to(device)
        lateral_correlations = torch.rand((sheet_size**2, 1, sheet_size, sheet_size), device=device)
        lateral_correlations /= lateral_correlations.sum([2,3], keepdim=True)
        self.lateral_correlations = lateral_correlations
        
        #masks = generate_gaussians(sheet_size, sheet_size, std_e).to(device)
        self.masks = \
            1 - self.untuned_inh/self.untuned_inh.view(self.untuned_inh.shape[0], -1).max(1)[0][:,None,None,None]
        
        
        # Homeostatic thresholds for each neuron
        self.thresholds = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.mean_activations = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        
        self.current_response = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        
        self.iterations = 16
        
        self.lateral_strength = 2
                
        self.homeo_target = 0.05
        self.homeo_lr = 1e-3
        self.bias_strength = 0.3
        
        self.afferent_unlearning = self.bias_strength
        self.lateral_unlearning = 0.23
        
        self.response_tracker = torch.zeros(self.iterations, 1, sheet_size, sheet_size, device=device)
        
    
    def forward(self, input_crop):
        
        self.current_input = input_crop
        self.current_response *= 0
        # Input crop is expected to be a 4D tensor: [batch_size, channels, N, N]
        # Process input through afferent weights
        afferent = F.conv2d(input_crop, self.afferent_weights, padding='valid')
        self.current_afferent = afferent.view(self.current_response.shape)

        long_range_inhibition = self.lateral_correlations * self.masks
        long_range_inhibition /= long_range_inhibition.sum([2,3], keepdim=True) + 1e-11
    
        self.iterations = 16
        for i in range(self.iterations):
            
            self.eq = 0.4
            untuned_peak = 1
            inversion = i < self.iterations//3
            
            self.t = self.eq * (1 - inversion)
            self.u = untuned_peak * inversion + (1 - self.eq)
            
            lateral_weights = \
                self.lateral_weights_exc - self.t*long_range_inhibition - self.u*self.untuned_inh
            
            lateral = F.conv2d(self.current_response, lateral_weights, padding='valid')
            lateral = lateral.view(self.current_response.shape)
                        
            # Initial activation
            z = self.current_afferent + lateral*self.lateral_strength - self.thresholds 
            
            self.current_response = torch.tanh(F.relu(z))
            
            self.response_tracker[i] = self.current_response.clone()
                        
                                    
        nonzero_act = self.current_response[self.current_response>0]
        if nonzero_act.shape[0]>0:
            self.lateral_strength -= (nonzero_act.mean() - 1/2)*1e-1
        self.mean_activations = self.mean_activations*(1-self.homeo_lr) + self.current_response*self.homeo_lr
        self.thresholds -= (self.homeo_target - self.mean_activations)*self.homeo_lr
            
            
    def hebbian_step(self, hebbian_lr):
                
        afferent_contributions = self.current_input + self.retinotopic_bias*self.bias_strength - self.afferent_unlearning
        delta_afferent = self.current_response.view(-1,1,1,1)*afferent_contributions

        self.afferent_weights += hebbian_lr*delta_afferent
        self.afferent_weights = torch.relu(self.afferent_weights)
        self.afferent_weights /= self.afferent_weights.sum([2,3], keepdim=True) + 1e-11
        
        lateral_contributions = self.current_response + self.lateral_bias*self.bias_strength - self.lateral_unlearning
        delta_lateral = self.current_response.view(-1,1,1,1)*lateral_contributions
        self.lateral_correlations += hebbian_lr*delta_lateral
        self.lateral_correlations = torch.relu(self.lateral_correlations)
        self.lateral_correlations /= self.lateral_correlations.sum([2,3], keepdim=True) + 1e-11
            

