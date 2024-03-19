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
        
        # Afferent (receptive field) weights for each neuron in the sheet
        afferent_weights = torch.rand((sheet_size**2, 1, input_size, input_size), device=device)
        afferent_weights += retinotopic_bias
        afferent_weights /= afferent_weights.sum([2,3], keepdim=True)
        self.afferent_weights = afferent_weights
        
        lateral_weights_exc = generate_gaussians(sheet_size, sheet_size, std_e).to(device)
        lateral_weights_exc /= lateral_weights_exc.sum([2,3], keepdim=True)
        self.lateral_weights_exc = lateral_weights_exc
        
        #lateral_bias = generate_gaussians(sheet_size, sheet_size, std_e*2+1).to(device)
        
        #lateral_correlations = generate_gaussians(sheet_size, sheet_size, 2*std_e+1).to(device)
        lateral_correlations = torch.rand((sheet_size**2, 1, sheet_size, sheet_size), device=device)
        lateral_correlations /= lateral_correlations.sum([2,3], keepdim=True)
        self.lateral_correlations = lateral_correlations
        
        self.masks = 1 - generate_gaussians(sheet_size, sheet_size, 0.1).to(device)
        
        # Homeostatic thresholds for each neuron
        self.thresholds = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        self.mean_activations = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        
        self.current_response = torch.zeros(1, 1, sheet_size, sheet_size, device=device)
        
        self.iterations = 20
        
        self.lateral_strength = 2
        
        self.hebbian_lr = 1e-3
        
        self.homeo_target = 0.02
        self.homeo_lr = 1e-3
        self.bias_strength = 4e-1
        
        self.afferent_unlearning = self.bias_strength
        self.lateral_unlearning = self.afferent_unlearning/10
        
    
    def forward(self, input_crop):
        
        self.current_input = input_crop
        self.current_response *= 0
        # Input crop is expected to be a 4D tensor: [batch_size, channels, N, N]
        # Process input through afferent weights
        afferent = F.conv2d(input_crop, self.afferent_weights, padding='valid')
        self.current_afferent = afferent.view(self.current_response.shape)
        
        long_range_inhibition = self.lateral_correlations*self.masks
        long_range_inhibition /= long_range_inhibition.sum([2,3], keepdim=True) + 1e-11
        
        lateral_weights = self.lateral_weights_exc - long_range_inhibition
        
        for i in range(self.iterations):
            
            lateral = F.conv2d(self.current_response, lateral_weights, padding='valid')
            lateral = lateral.view(self.current_response.shape)
                        
            # Initial activation
            z = self.current_afferent + lateral*self.lateral_strength - self.thresholds 
            
            self.current_response = torch.tanh(F.relu(z))
                        
                                    
        nonzero_act = self.current_response[self.current_response>0]
        if nonzero_act.shape[0]>0 and False:
            self.lateral_strength -= (nonzero_act.mean() - 1/2)*1e-1
        self.mean_activations = self.mean_activations*(1-self.homeo_lr) + self.current_response*self.homeo_lr
        self.thresholds -= (self.homeo_target - self.mean_activations)*self.homeo_lr
            
            
    def hebbian_step(self):
                
        afferent_contributions = self.current_input + self.retinotopic_bias*self.bias_strength - self.afferent_unlearning
        delta_afferent = self.current_response.view(-1,1,1,1)*afferent_contributions

        self.afferent_weights += self.hebbian_lr*delta_afferent
        self.afferent_weights = torch.relu(self.afferent_weights)
        self.afferent_weights /= self.afferent_weights.sum([2,3], keepdim=True) + 1e-11
        
        lateral_contributions = self.current_response - self.lateral_unlearning
        delta_lateral = self.current_response.view(-1,1,1,1)*lateral_contributions
        self.lateral_correlations += self.hebbian_lr*delta_lateral
        self.lateral_correlations = torch.relu(self.lateral_correlations)
        self.lateral_correlations /= self.lateral_correlations.sum([2,3], keepdim=True) + 1e-11
            

