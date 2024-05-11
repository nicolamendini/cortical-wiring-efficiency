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
import matplotlib.pyplot as plt
import cv2
import matplotlib.animation as animation
from IPython.display import HTML

import nn_template 

class RandomCropDataset(Dataset):
    def __init__(self, directory, crop_size):
        self.directory = directory
        self.crop_size = crop_size
        self.images = [os.path.join(directory, f) for f in os.listdir(directory)
                       if os.path.isfile(os.path.join(directory, f))]
        # Pre-filter images smaller than the crop size
        self.images = [img for img in self.images if self._image_size(img) >= crop_size]

    def _image_size(self, filepath):
        with Image.open(filepath) as img:
            return min(img.size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path)  # Open with PIL
        
        # Apply random transformation
        transformed_image = self.random_transformation(image, self.crop_size)
        return transformed_image

    def random_transformation(self, image, N):
        # Random rotation degrees
        rotation_degrees = random.uniform(-180, 180)
        
        # Transformation pipeline without resizing
        transformation = transforms.Compose([
            transforms.RandomRotation((rotation_degrees, rotation_degrees)),
            transforms.CenterCrop(N),  # Crop to NxN from the center
            transforms.ToTensor(),  # Convert to tensor
        ])
        
        transformed_image = transformation(image)
        return transformed_image

def create_dataloader(root_dir, crop_size, batch_size, num_workers):
    dataset = RandomCropDataset(root_dir, crop_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return dataloader

def generate_gaussians(number_of_gaussians, size_of_gaussian, sigma):
    # Create a grid of coordinates (x, y) for the centers
    lin_centers = torch.linspace(-number_of_gaussians / 2, number_of_gaussians / 2, number_of_gaussians)

    if number_of_gaussians==1:
        lin_centers += 1/2
        
    x_centers, y_centers = torch.meshgrid(lin_centers, lin_centers, indexing='ij')
    
    # Create a grid of coordinates (x, y) for a single Gaussian of size MxM
    lin_gaussian = torch.linspace(-size_of_gaussian / 2, size_of_gaussian / 2, size_of_gaussian)
    x_gaussian, y_gaussian = torch.meshgrid(lin_gaussian, lin_gaussian, indexing='ij')
    
    # Flatten the center coordinates to easily use broadcasting
    x_centers_flat = x_centers.reshape(-1, 1, 1)
    y_centers_flat = y_centers.reshape(-1, 1, 1)
    
    # Calculate the squared distance for each Gaussian center to each point in the MxM grid
    dist_squared = (x_gaussian - x_centers_flat) ** 2 + (y_gaussian - y_centers_flat) ** 2
    
    # Precompute the Gaussian denominator
    gaussian_denom = 2 * sigma ** 2
    
    # Calculate the Gaussians
    gaussians = torch.exp(-dist_squared / gaussian_denom)
    
    # Normalize each Gaussian to have a maximum value of 1
    gaussians /= gaussians.view(number_of_gaussians**2, -1).sum(1).unsqueeze(1).unsqueeze(1)
    
    # Reshape to have each Gaussian in its own channel (N*N, M, M)
    gaussians = gaussians.reshape(number_of_gaussians**2, 1, size_of_gaussian, size_of_gaussian)
    
    return gaussians


def generate_circles(number_of_circles, size_of_circles, radius=0):
    # Create a grid of coordinates (x, y) for the centers
    lin_centers = torch.linspace(-number_of_circles / 2, number_of_circles / 2, number_of_circles)

    if number_of_circles==1:
        lin_centers += 1/2
        
    x_centers, y_centers = torch.meshgrid(lin_centers, lin_centers, indexing='ij')
    
    # Create a grid of coordinates (x, y) for a single Gaussian of size MxM
    lin_circle = torch.linspace(-size_of_circles / 2, size_of_circles / 2, size_of_circles)
    x_circle, y_circle = torch.meshgrid(lin_circle, lin_circle, indexing='ij')
    
    # Flatten the center coordinates to easily use broadcasting
    x_centers_flat = x_centers.reshape(-1, 1, 1)
    y_centers_flat = y_centers.reshape(-1, 1, 1)
    
    # Calculate the squared distance for each Gaussian center to each point in the MxM grid
    dist_squared = (x_circle - x_centers_flat) ** 2 + (y_circle - y_centers_flat) ** 2
    
    circles = dist_squared < ((radius + 1)**2)

    circles = circles.reshape(number_of_circles**2, 1, size_of_circles, size_of_circles).float()
    
    return circles
    

def get_detectors(gabor_size, discreteness, device='cuda'):
    orientations = torch.linspace(0, np.pi, discreteness, device=device)
    lambd = 12.0
    sigma = 4.0
    gamma = 1
    psi = torch.tensor([0, np.pi/2], device=device)

    # Create a meshgrid for Gabor function
    x, y = torch.meshgrid(torch.linspace(-gabor_size//2 + 1, gabor_size//2 + 1, gabor_size, device=device), 
                          torch.linspace(-gabor_size//2 + 1, gabor_size//2 + 1, gabor_size, device=device), indexing='ij')

    x = x.expand(discreteness, 2, gabor_size, gabor_size)
    y = y.expand(discreteness, 2, gabor_size, gabor_size)
    orientations = orientations.view(discreteness, 1, 1, 1)
    psi = psi.view(1, 2, 1, 1)

    x_theta = x * torch.cos(orientations) + y * torch.sin(orientations)
    y_theta = -x * torch.sin(orientations) + y * torch.cos(orientations)
    
    gb = torch.exp(-.5 * (x_theta**2 + gamma**2 * y_theta**2) / sigma**2) * torch.cos(2 * np.pi * x_theta / lambd + psi)
    
    return gb  # (discreteness, 2, gabor_size, gabor_size)

def get_orientations(weights, discreteness=30, gabor_size=21):
    
    device = weights.device

    #weights = weights[:, 0] - weights[:, 1]
    #weights = weights[:, None]
    
    # input is (M, 1, S, S)
    M, _, S, _ = weights.shape
    detectors = get_detectors(gabor_size, discreteness).to(device)
    
    # Prepare weights and detectors for convolution
    responses = torch.zeros((M, discreteness, 2, S, S), device=device)

    # Convolution over the receptive fields with each detector
    for i in range(discreteness):
        for j in range(2):
            responses[:, i, j] = F.conv2d(weights, detectors[i:i+1, j].unsqueeze(1), padding='valid').squeeze(1)
            
            
    responses = responses.view(M, discreteness, 2, -1).max(3)[0]
                
    # Compute phase map and magnitude of responses
    magnitudes = torch.sqrt((responses**2).sum(dim=2))
    orientations = magnitudes.max(dim=1)[1]
    max_responses = magnitudes.max(dim=1)[0]
    
    phases = responses.gather(1, orientations[:,None,None].expand(-1,-1,2))
    phase_map = torch.atan2(phases[:, :, 1], phases[:, :, 0])

    orientation_map = orientations.float() / discreteness * np.pi
    orientation_map = orientation_map % torch.pi/2
    
    # output is M
    return orientation_map, phase_map
    

def get_grids(W, H, kernel_size, N, device='cuda'):

    # Generate grid positions for each patch using broadcasting
    grid_positions_w = torch.linspace(0, W - kernel_size, N, device=device).view(-1, 1) / (W - 1) * 2 - 1
    grid_positions_h = torch.linspace(0, H - kernel_size, N, device=device).view(1, -1) / (H - 1) * 2 - 1
    
    # Compute normalized coordinates for each patch
    x = grid_positions_w + torch.linspace(0, kernel_size - 1, kernel_size, device=device).view(1, -1) / (W - 1) * 2
    y = grid_positions_h + torch.linspace(0, kernel_size - 1, kernel_size, device=device).view(-1, 1) / (H - 1) * 2
    
    # Stack and reshape to create grid
    grids_x, grids_y = torch.meshgrid(x.flatten(), y.flatten())
    grids = torch.stack((grids_x, grids_y), dim=-1)
    grids = grids.view(N, kernel_size, kernel_size, N,  2).permute(3,0,2,1,4).reshape(N*N, kernel_size, kernel_size, 2)

    return grids

def extract_patches(input_image, grids):
    """
    Extracts N patches of size kernel_size x kernel_size from input_image, calculating the step size automatically.
    This function now handles cases where the last patch may not fit perfectly and returns patches with dimensions matching the input.
    This version leverages CUDA for improved performance.
    """
    Nxx2 = grids.shape[0]
    # Extract patches
    patches = F.grid_sample(input_image.expand(Nxx2, -1, -1, -1), grids, mode='bilinear', align_corners=False)
    
    return patches

def init_nn(input_size, output_size, device='cuda'):

    network = {}
    
    network['structure'] = [
        ('flatten', 1, input_size**2),
        ('dense', output_size**2, input_size**2),
        #('dense', output_size**2, output_size**2),
        #('dense', output_size**2, output_size**2),
        ('unflatten', 1, output_size)
    ]
    
    network['model'] = nn_template.Network(network['structure'], device=device)
    params_list = [list(network['model'].layers[l].parameters()) for l in range(len(network['structure']))]
    params_list = sum(params_list, [])
    
    network['optim'] = torch.optim.Adam(params_list, lr=1e-3)
    network['activ'] = torch.relu
    
    return network

def nn_loss(network, true_input, reco_input, loss_weights):

    mse = ((true_input - reco_input)**2).sum([1,2,3]) * loss_weights
    mse = mse.sum()
    l1 = sum(
        [
            list(network['model'].layers[l].parameters())[0].abs().sum() \
            for l in range(1,len(network['structure'])-1)
        ]
    )
    l1 = l1.sum()
    loss = mse + 0*l1
    return loss

# Function to compute the Laplacian Of Gaussian Operator
def get_log(size, std):
    
    distance = torch.arange(size) - size//2
    x = distance.expand(1,1,size,size)**2
    y = x.transpose(-1,-2)
    t = (x + y) / (2*std**2)
    LoG = -1/(np.pi*std**2) * (1-t) * torch.exp(-t)
    LoG = LoG - LoG.mean()
    return LoG

def oddenise(number):
    return round(number)+1 if round(number)%2==0 else round(number)


def get_gaussian(size, std, yscale=1, centre_x=0, centre_y=0):
    
    distance = torch.arange(size) - size//2 - centre_x*(size//2)
    x = distance.expand(1,1,size,size)**2
    distance = torch.arange(size) - size//2 - centre_y*(size//2)
    y = (distance.expand(1,1,size,size)**2).transpose(-1,-2)*yscale
    t = (x + y) / (2*std**2)
    gaussian = 1 / (np.sqrt(2*np.pi*std**2)) * torch.exp(-t)
    gaussian /= gaussian.sum()
        
    return gaussian 

    

