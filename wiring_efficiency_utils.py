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
from sklearn.decomposition import PCA
import umap
import matplotlib.cm as cm

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
            transforms.RandomRotation((rotation_degrees, rotation_degrees), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(N),  # Crop to NxN from the center
            transforms.ToTensor(),  # Convert to tensor
        ])
        
        transformed_image = transformation(image)
        return transformed_image

def create_dataloader(root_dir, crop_size, batch_size, num_workers):
    dataset = RandomCropDataset(root_dir, crop_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return dataloader

def generate_gaussians(number_of_gaussians, size_of_gaussian, sigma, offset=False):
    # Create a grid of coordinates (x, y) for the centers
    lin_centers = torch.linspace(-size_of_gaussian / 2, size_of_gaussian / 2, number_of_gaussians)
        
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

    if number_of_circles == 1:
        lin_centers += 0.5  # Adjust the center for a single circle

    x_centers, y_centers = torch.meshgrid(lin_centers, lin_centers, indexing='ij')
    
    # Create a grid of coordinates (x, y) for a single circle of size MxM
    lin_circle = torch.linspace(-size_of_circles / 2, size_of_circles / 2, size_of_circles)
    x_circle, y_circle = torch.meshgrid(lin_circle, lin_circle, indexing='ij')
    
    # Flatten the center coordinates to easily use broadcasting
    x_centers_flat = x_centers.reshape(-1, 1, 1)
    y_centers_flat = y_centers.reshape(-1, 1, 1)
    
    # Calculate the squared distance from each circle center to each point in the MxM grid
    dist_squared = (x_circle - x_centers_flat) ** 2 + (y_circle - y_centers_flat) ** 2
    dist = torch.sqrt(dist_squared)
    
    # Calculate circle membership with smoothing
    # Define the radius band for smooth transition (0.5 pixel width)
    radius_inner = radius - 0.5
    radius_outer = radius + 0.5

    # Compute a smooth transition in pixel values across the boundary of the circle
    circles = 1 - torch.clamp((dist - radius_inner) / (radius_outer - radius_inner), 0, 1)

    # Reshape back to the format (number_of_circles^2, 1, size_of_circles, size_of_circles)
    circles = circles.reshape(number_of_circles**2, 1, size_of_circles, size_of_circles)
    
    return circles.float()
    

def get_detectors(gabor_size, discreteness, device='cuda'):
    orientations = torch.linspace(0, np.pi, discreteness, device=device)
    lambd = gabor_size
    sigma = gabor_size/5
    gamma = 1
    psi = torch.tensor([0, np.pi/2], device=device)

    # Create a meshgrid for Gabor function
    x, y = torch.meshgrid(torch.linspace(-gabor_size//2 + 1/2, gabor_size//2 + 1/2, gabor_size, device=device), 
                          torch.linspace(-gabor_size//2 + 1/2, gabor_size//2 + 1/2, gabor_size, device=device), indexing='ij')

    x = x.expand(discreteness, 2, gabor_size, gabor_size)
    y = y.expand(discreteness, 2, gabor_size, gabor_size)
    orientations = orientations.view(discreteness, 1, 1, 1)
    psi = psi.view(1, 2, 1, 1)

    x_theta = x * torch.cos(orientations) + y * torch.sin(orientations)
    y_theta = -x * torch.sin(orientations) + y * torch.cos(orientations)
    
    gb = torch.cos(2 * np.pi * x_theta / lambd + psi) #* torch.exp(-.5 * (x_theta**2 + gamma**2 * y_theta**2) / sigma**2)
    
    gb = gb * get_circle(gabor_size, gabor_size/2, smooth=True).cuda()
    
    gb -= gb.mean([-1,-2], keepdim=True)
    
    return gb  # (discreteness, 2, gabor_size, gabor_size)

def get_orientations(weights, discreteness=101, gabor_size=25):
    
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
    
    phases = responses.gather(1, orientations[:,None,None].expand(-1,-1,2))
    phase_map = torch.atan2(phases[:, :, 1], phases[:, :, 0])
    
    shifts = torch.arange(discreteness)[None] + orientations[:,None].cpu() + discreteness//2
    shifts = shifts % discreteness
    mean_tc = magnitudes.cpu().gather(1, shifts).mean(0)
    
    orientation_map = orientations.float() / discreteness * np.pi
    orientation_map = orientation_map % torch.pi/2
    
    # output is M
    return orientation_map, phase_map, mean_tc
    

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
        ('relu', ),
        ('dense', output_size**2, output_size**2),
        ('unflatten', 1, output_size)
    ]
    
    network['model'] = nn_template.Network(network['structure'], device=device)
    params_list = [list(network['model'].layers[l].parameters()) for l in range(len(network['structure']))]
    params_list = sum(params_list, [])
    
    network['optim'] = torch.optim.Adam(params_list, lr=2e-3)
    network['activ'] = torch.sigmoid
    
    return network

def nn_loss(network, true_input, reco_input):

    mse = ((true_input - reco_input)**2).mean([1,2,3])
    #mse = mse.sum()
    #l1 = sum(
    #    [
    #        list(network['model'].layers[l].parameters())[0].abs().sum() \
    #        for l in range(1,len(network['structure'])-1)
    #    ]
    #)
    #l1 = l1.sum()
    #loss = mse + 0*l1
    
    #bce = true_input * torch.log(reco_input + 1e-11) + (1-true_input) * torch.log(1 - reco_input + 1e-11)
    #bce = - bce.mean([1,2,3])
    
    loss = mse.mean()
    loss_std = mse.std()
    
    #loss = F.binary_cross_entropy_with_logits(reco_input, true_input, reduction='sum')
    
    return loss, loss_std

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

def get_spectral_entropy(codes):
    
    fft = (torch.fft.fft2(codes).abs()**2).mean(0, keepdim=True)
    #spectral_dist = fft / (fft.sum([1,2,3], keepdim=True) + 1e-11)
    #spectral_entropy = spectral_dist * torch.log(spectral_dist + 1e-11)
    #spectral_entropy = - spectral_entropy.sum()
        
    return fft.sum()

def count_significant_freqs(codes):
    # Assuming codes is of shape (N, H, W), where N is the batch size

    # Step 0: Perform 2D FFT on each image in the batch
    fourier_spectrum = torch.fft.fft2(codes[:,0])
    fourier_spectrum[:,0,0] = 0

    # Step 1: Calculate the magnitude of the complex numbers
    magnitudes = torch.abs(fourier_spectrum)
    
    # Step 2: Calculate the squared magnitudes (power of each component)
    power = magnitudes ** 2
    
    # Step 3: Sum to find total power for each sample in the batch
    total_power = torch.sum(power, dim=(1, 2))  # Sum over both H and W dimensions
    
    # Step 4: Flatten the powers for sorting, then sort powers in descending order for each example
    sorted_power = torch.sort(power.view(power.size(0), -1), dim=1, descending=True)[0]
    
    # Step 5: Calculate cumulative sum of the sorted powers along each batch
    cumulative_power = torch.cumsum(sorted_power, dim=1)
        
    counts = cumulative_power > (total_power[:,None] * 0.95)
        
    # The number of components needed to reach 95% of the total power
    return counts.float().sum(1).mean()


# Function to measure the typical distance between iso oriented map domains
# Samples a certain number of orientations given by 'precision' and returns 
# the histograms of the gaussian doughnuts that were used to fit the curve together with the peak
def get_typical_dist_fourier(orientations, border_cut, precision=10, mask=1, match_std=1):
    
    # R is the size of the map after removing some padding size, must be odd thus 1 is subtractedS
    grid_size = orientations.shape[-1] - 1
    R = (grid_size - border_cut*2)

    spectrum = 0
    avg_spectrum = torch.zeros(R,R)
    avg_peak = 0
    avg_hist = torch.zeros(R//2)
    ang_range = torch.linspace(0, torch.pi-torch.pi/precision, precision)

    # average over a number of rings, given by precision
    for i in range(precision):
        
        # compute the cosine similarity and subtract that of the opposite angle
        # this is needed to get a cleaner ring
        output = torch.cos(orientations - ang_range[i])**2
        output -= torch.cos(orientations - ang_range[i] + torch.pi/2)**2
        spectrum = output[border_cut:-(border_cut+1),border_cut:-(border_cut+1)].cpu()
            
        #plt.imshow(spectrum)
        #plt.show()
            
        # compute the fft and mask it to remove the central bias
        af = torch.fft.fft2(spectrum)
        af = torch.abs(torch.fft.fftshift(af))
        af *= ~get_circle(af.shape[-1], mask)[0,0]
        
        hist, peak_interpolate = match_ring(af, match_std)
            
        # add the results to the average trackers
        # 1/peak_interpolate is to convert from freq to wavelength
        avg_peak += 1/peak_interpolate
        avg_spectrum += af
        avg_hist += hist
        
    avg_peak /= precision
    avg_spectrum /= precision
    avg_hist /= precision
    
    return avg_peak, avg_spectrum, avg_hist

# function to find the peak of a fourier transform
def match_ring(af, match_std=1):
    
    R = af.shape[-1]
    hist = torch.zeros(R//2)
    steps = torch.fft.fftfreq(R)
    
    # use progressively bigger doughnut funtions to find the most active radius
    # which will correspond to the predominant frequency
    for r in range(R//2):

        doughnut = get_doughnut(R, r, match_std)
        prod = af * doughnut
        hist[r] = (prod).sum()

    argmax_p = hist.argmax()
    peak_interpolate = steps[argmax_p]

    # interpolate between the peak value and its two neighbours to get a more accurate estimate
    if argmax_p+1<R//2:
        base = hist[argmax_p-2]
        a,b,c = (hist[argmax_p-1]-base).abs(), (hist[argmax_p]-base).abs(), (hist[argmax_p+1]-base).abs()
        tot = a+b+c
        a /= tot
        b /= tot
        c /= tot
        peak_interpolate = a*steps[argmax_p-1] + b*steps[argmax_p] + c*steps[argmax_p+1]
            
    return hist, peak_interpolate

# Function to get a doughnut function which is 
# defined as a Gaussian Ring around a radius value with a certain STD
def get_doughnut(size, r, std):
    
    distance = torch.arange(size) - size//2
    x = distance.expand(size,size)**2
    y = x.transpose(-1,-2)
    t = torch.sqrt(x + y)
    doughnut = torch.exp(-(t - r)**2/(2*std**2))
    doughnut /= doughnut.sum()
    return doughnut

# Function to get a circle mask with ones inside and zeros outside
def get_circle(size, radius, smooth=True):
    
    distance = torch.arange(size) - size//2
    x = distance.expand(1,1,size,size)**2
    y = x.transpose(-1,-2)
    circle = torch.sqrt(x + y)
    
    # Calculate circle membership with smoothing
    # Define the radius band for smooth transition (0.5 pixel width)
    radius_inner = radius - 0.5
    radius_outer = radius + 0.5

    if smooth:
        # Compute a smooth transition in pixel values across the boundary of the circle
        circle = 1 - torch.clamp((circle - radius_inner) / (radius_outer - radius_inner), 0, 1)
        
    else:
        circle = circle < radius
    
    return circle


def get_effective_dims(code_tracker, debug=False):
    
    clean_codes = [code for code in code_tracker if code.sum()]
    codes_var = torch.cat(clean_codes[-10000:], dim=0)
    
    code_size = codes_var.shape[-1]
    
    fft1 = torch.fft.fft2(codes_var) - torch.fft.fft2(TF.rotate(codes_var, 5))
    fft1 = torch.fft.fftshift(fft1)
    #fft1 *= ~get_circle(code_size, 1).cuda()
    fft1 *= get_circle(code_size, (code_size/2)).cuda()

    meanfft = (fft1.abs()**2).mean([0,1]).cpu()
    meanfft /= meanfft.sum()
    sorted_power = torch.argsort(meanfft.view(-1), dim=0, descending=True)
    ordered = (meanfft.view(-1)[sorted_power]).cumsum(dim=0) < .75
    cutoff = ordered.sum()
    
    plotfft = meanfft.view(-1)
    plotfft[sorted_power[cutoff:]] = 0
    plotfft = plotfft.view(code_size,code_size)

    fft1 = fft1.view(-1, code_size**2)
    fft1[:, sorted_power[cutoff:]] = 0
    fft1 = fft1.view(-1, 1, code_size, code_size)
    fft1 = torch.fft.fftshift(fft1)
    
    _, peak = match_ring(plotfft, 1)
    
    if debug:
        
        reco = torch.relu(torch.fft.ifft2(fft1).float())
        plt.imshow(codes_var[0,0].cpu())
        plt.show()
        plt.imshow(plotfft, cmap=cm.Greys_r)
        plt.show()
        plt.imshow(reco[0,0].cpu())
        plt.show()
        
        cos = cosim(codes_var, reco)

        print(cos, cutoff)
        
    return cutoff, plotfft, peak

def cosim(X, Y, weighted=False):
    
    cos = X.cpu() * Y.cpu()
    norm = torch.sqrt((X.cpu()**2).sum([2,3], keepdim=True) * (Y**2).cpu().sum([2,3], keepdim=True))
    cos /= norm + 1e-11
    cos = cos.sum([2,3])
    
    if weighted:
        weights = X.sum([2,3])
        weights /= weights.sum()
        cos = (cos * weights).sum()
        
    else:
        cos = cos.mean()
    
    return cos    

def get_pca_dimensions(code_tracker, n_samps):
        
    clean_codes = [code for code in code_tracker if code.sum()]
    codes_var = torch.cat(clean_codes[-10000:], dim=0)
    
    size = codes_var.shape[-1]
    
    pca = PCA(n_components=size**2)
    pca.fit(codes_var.cpu().view(-1,size**2))
    eff_dim = (pca.explained_variance_ratio_.cumsum() < 0.95).sum()
    
    comp_sampled = torch.tensor(pca.components_).view(size,size,size**2)
    step = size // n_samps
    comp_sampled = comp_sampled[::step, ::step][:n_samps, :n_samps]
    
    comp_sampled = comp_sampled.view(n_samps, n_samps, size, size)
    comp_sampled = comp_sampled.permute(0,2,1,3).reshape(n_samps*size, n_samps*size)
        
    return eff_dim, comp_sampled

def get_umap(code_tracker, window_size):
    
    codes_var = torch.cat(code_tracker, dim=0).cpu()   
    codes_var = F.unfold(codes_var, window_size, stride=window_size).permute(0,2,1).reshape(-1, 1, window_size, window_size)
    #sorting_idx = codes_var.sum([1,2,3]).sort(descending=True)[1]
    #codes_var = codes_var[sorting_idx][:2000]
    codes_var = codes_var.reshape(-1, window_size**2)
        
    size = codes_var.shape[-1]
    
    pca = PCA(n_components=6)
    pca.fit(codes_var.cpu().view(-1,window_size**2))
    
    print('variance explained: ', pca.explained_variance_ratio_.sum())
    
    reduced = pca.transform(codes_var)
    
    embedding = umap.UMAP(
                n_neighbors=1000,
                min_dist=0.8,
                metric='cosine',
                n_components=3,
                init='spectral'
             ).fit_transform(reduced)
    
    return embedding, codes_var
    

# function to plot the umap
def draw_umap(embedding, size, dims=3, title='3D projection'):
    plt.ioff()
    plt.title(title, fontsize=18)
    fig = plt.figure(figsize=(size, size))
    if dims==3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embedding[:,0], embedding[:,1], embedding[:,2], s=10)
    else:
        ax = fig.add_subplot(111)
        ax.scatter(embedding[:,0], embedding[:,1], s=10)
    plt.show()
    return fig, ax


def select_grid_modules(code_tracker, window_size):
    
    codes_var = torch.cat(code_tracker, dim=0).cpu()
    #sorting_idx = codes_var.sum([1,2,3]).sort(descending=True)[1]
    #codes_var = codes_var[sorting_idx][2000:3000]    
    codes_var = F.unfold(codes_var, window_size, stride=window_size)
    codes_var = codes_var.permute(0,2,1).reshape(-1, 1, window_size, window_size)
    
    autocorrelograms = compute_autocorrelograms(codes_var, 3, 0.7, 0)
    
    embedding = umap.UMAP(
                n_neighbors=5,
                min_dist=0.05,
                metric='manhattan',
                n_components=2,
                init='spectral'
             ).fit_transform(autocorrelograms.view(-1, window_size**2))
    
    return embedding, autocorrelograms


def get_gratings(size, orientation, period, phase):
    """
    Generates a sinusoidal grating.
    
    Arguments:
    size : int - The size of the grating image (height and width).
    orientation : float - The orientation of the grating in degrees.
    period : int - The spatial period of the grating, in pixels.
    phase : float - The phase shift of the sinusoidal pattern, in degrees.
    
    Returns:
    torch.Tensor - A 2D tensor representing the grating pattern.
    """
    # Convert orientation and phase from degrees to radians
    orientation = torch.deg2rad(torch.tensor(orientation))
    phase = torch.deg2rad(torch.tensor(phase))
    
    # Create a grid of coordinates
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    
    # Adjust the coordinates based on the center of the image
    x = x - size // 2
    y = y - size // 2
    
    # Rotate the coordinate system by the orientation
    x_rot = x * torch.cos(orientation) + y * torch.sin(orientation)
    y_rot = -x * torch.sin(orientation) + y * torch.cos(orientation)
    
    # Apply the sinusoidal function
    grating = torch.sin(2 * torch.pi * x_rot / period + phase)
    
    return grating


def compute_autocorrelograms(tiles, r, sigma, eps):
    """
    Computes spatial autocorrelograms for a batch of tiles, applies a mask around the peak,
    smooths the result with a Gaussian filter, and thresholds the autocorrelogram.

    Parameters:
    tiles (torch.Tensor): Input tensor of shape (N, 1, W, W)
    r (int): Radius for masking around the peak
    sigma (float): Standard deviation for Gaussian smoothing
    eps (float): Threshold for final autocorrelogram

    Returns:
    torch.Tensor: Processed autocorrelograms of shape (N, 1, W, W)
    """
    N, C, W, _ = tiles.shape
    gauss = get_gaussian(oddenise(sigma*6), sigma)

    # Step 1: Compute autocorrelograms
    # Need to convert each tile to a full batch where each tile is a kernel
    result = torch.empty((N, C, W, W))
    for i in range(N):
        # Applying convolution for each tile with itself
        result[i] = F.conv2d(tiles[i].unsqueeze(0), tiles[i].unsqueeze(0), padding=W//2)

    # Normalize the autocorrelograms (optional but typical)
    result /= result.view(N, -1).max(1, keepdim=True)[0].view(N, 1, 1, 1)

    # Step 2: Locate the peaks
    # Assume peak is at the center for autocorrelation (common assumption, but you might need to adjust)
    peak_coords = (W//2, W//2)

    # Step 3: Mask out pixels within a radius 'r' from the peak
    for i in range(N):
        y, x = torch.meshgrid(torch.arange(W), torch.arange(W), indexing='ij')
        mask = ((x - peak_coords[1])**2 + (y - peak_coords[0])**2) <= r**2
        result[i, 0, mask] = 0
        mask = ((x - peak_coords[1])**2 + (y - peak_coords[0])**2) >= (W/2)**2
        result[i, 0, mask] = 0

    # Step 4: Apply Gaussian smoothing
    if sigma > 0:
        for i in range(N):
            result[i] = F.conv2d(result[i], gauss, padding=gauss.shape[-1]//2)

    # Step 5: Threshold the autocorrelograms
    result = torch.where(result >= eps, result, torch.zeros_like(result))

    return result
    
    


    

