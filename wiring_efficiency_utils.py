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

def get_detectors(gabor_size, discreteness, device='cuda'):
    orientations = torch.linspace(0, np.pi, discreteness, device=device)
    lambd = 6.0
    sigma = 4.0
    gamma = 0.8
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

def get_orientations(weights, discreteness=30, gabor_size=27):
    
    device = weights.device
    
    # input is (M, 1, S, S)
    M, _, S, _ = weights.shape
    detectors = get_detectors(gabor_size, discreteness).to(device)
    
    # Prepare weights and detectors for convolution
    responses = torch.zeros((M, discreteness, 2, S, S), device=device)

    # Convolution over the receptive fields with each detector
    for i in range(discreteness):
        for j in range(2):
            responses[:, i, j] = F.conv2d(weights, detectors[i:i+1, j].unsqueeze(1), padding='same').squeeze(1)
            
            
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


def show_map(model, batch, random_sample=None):
    
    plt.figure(figsize=(12, 18))
    
    if not random_sample:
        random_sample = random.randint(0, model.afferent_weights.shape[0] - 1)
        
    random_batch = random.randint(0, batch.shape[0] - 1)
    
    model(batch[random_batch][None])
    
    titles = [
        "Current Input", "Afferent Weights", "Lateral Correlations",
        "Lateral Weights Exc", "Current Response", "Current Response Histogram",
        "Orientation Map", "Orientation Histogram", "Phase Map"
    ]

    # Displaying the model's current input
    plt.subplot(4, 3, 1)
    plt.imshow(model.current_input[0, 0].detach().cpu())
    plt.title(titles[0])

    # Afferent weights of a random sample
    plt.subplot(4, 3, 2)
    plt.imshow(model.afferent_weights[random_sample, 0].detach().cpu())
    plt.title(titles[1])

    # Lateral correlations of the random sample
    plt.subplot(4, 3, 3)
    plotvar = model.lateral_correlations[random_sample, 0] * model.masks[random_sample, 0]
    plotvar = plotvar*model.eq + model.untuned_inh[random_sample, 0]*(1-model.eq)
    plt.imshow(plotvar.detach().cpu())
    plt.title(titles[2])

    # Lateral weights excitation of the random sample
    plt.subplot(4, 3, 4)
    plt.imshow(model.lateral_weights_exc[random_sample, 0].detach().cpu())
    plt.title(titles[3])

    # Model's current response
    plt.subplot(4, 3, 5)
    plt.imshow(model.current_response[0, 0].detach().cpu())
    plt.title(titles[4])

    # Histogram of the current response
    plt.subplot(4, 3, 6)
    hist = model.current_response.flatten().detach().cpu().numpy()
    plt.hist(hist[hist > 0])
    plt.title(titles[5])

    # Generate and display orientation and phase maps
    weights = model.afferent_weights.clone()
    M = int(np.sqrt(model.afferent_weights.shape[0]))  # Assuming MxM grid for reshaping
    ori_map, phase_map = get_orientations(weights)
    ori_map = ori_map.reshape(M, M).cpu()
    phase_map = phase_map.reshape(M, M).cpu()
    
    # Orientation map
    plt.subplot(4, 3, 7)
    plt.imshow(ori_map, cmap='hsv')
    plt.title(titles[6])

    # Orientation histogram
    plt.subplot(4, 3, 8)
    plt.hist(ori_map.flatten())
    plt.title(titles[7])

    # Phase map
    plt.subplot(4, 3, 9)
    plt.imshow(phase_map, cmap='hsv')
    plt.title(titles[8])

    plt.show()
    
# Function to animate an array as a useful visualisation
def animate(array, n_frames, cmap=None, interval=300):
    
    fig = plt.figure(figsize=(6,6))
    global i
    i = -2
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    im = plt.imshow(array[0], animated=True, cmap=cmap)

    def updatefig(*args):
        global i
        if (i < n_frames - 1):  # ensure that we don't go out of bounds
            i += 1
        im.set_array(array[i])
        return im,

    anim = animation.FuncAnimation(fig, updatefig, frames=n_frames, interval=interval, repeat=True)
    plt.close(fig)  # Prevents the static plot from showing in the notebook
    return HTML(anim.to_jshtml())  # Directly display the animation