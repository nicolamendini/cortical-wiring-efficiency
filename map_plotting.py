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

from wiring_efficiency_utils import *

def sample_and_plot(distribution, num_samples, sample_idx, full=False):

    M = distribution.shape[-1]

    # Convert distribution to PyTorch tensor and flatten for sampling
    dist_tensor = torch.tensor(distribution.flatten(), dtype=torch.float)
    
    # Sample S locations from the distribution
    indices = torch.multinomial(dist_tensor, num_samples, replacement=False)

    if full:
        indices = torch.where(dist_tensor>0)[0]
        num_samples = indices.shape[0]
    
    # Convert flat indices back to 2D indices
    y, x = np.unravel_index(indices.numpy(), (M, M))
    
    # Get the center coordinates
    center_x = sample_idx % M
    center_y = sample_idx // M
    
    # Visualization with Matplotlib
    plt.figure(figsize=(12,12))
    plt.subplot(1,2,1)
    plt.imshow(distribution*0, cmap='binary', interpolation='nearest')
    
    # Add scatter to the sampled points with random scatter
    x_scatter = x + np.random.randn(num_samples) * 0.15  # Add random scatter to x coordinates
    y_scatter = y + np.random.randn(num_samples) * 0.15  # Add random scatter to y coordinates
    plt.scatter(x_scatter, y_scatter, color='black', s=10, alpha=0.5)  # Add transparency to the sampled points
    
    # Add scatter to the sampled points and draw lines from center to each point
    for i in range(len(x)):
        plt.plot([center_x, x[i]], [center_y, y[i]], color='black', linestyle='-', linewidth=0.5, alpha=0.)  # More transparent lines
    
    plt.scatter(center_x, center_y, color='black', s=100)  # Plot the center
    plt.title('Distribution with Sampled Locations')
    plt.axis('off')
    plt.subplot(1,2,2)
    # Load the image from local file
    image_path = "macaque_patches_v1.png"  # Change to your image filename
    image = Image.open(image_path)
    plt.title('Patchy Connectivity Macaque V1')
    # Display the rotated image
    plt.imshow(image)
    plt.axis('off')  # Turn off axis for the image

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


def show_map(model, network, random_sample=None):

    plt.figure(figsize=(12, 14))
    titles = [
        "Current Input", "Afferent Weights", "Current Aff Response", "Exc - inh",
        "Lateral correlations", "Current Response", "Current Response Histogram",
        "Orientation Map", "Orientation Histogram", "Phase Map", "L4 afferent", "Map detector Gabor",
        "Reconstruction", "Positive Afferent", "Thresholds", "L4 response"
    ]

    # Displaying the model's current input
    img = model.current_input[0, 0].detach().cpu()
    img[0,0] = 1
    plt.subplot(4, 4, 1)
    plt.imshow(img)
    plt.title(titles[0])

    # Afferent weights of a random sample
    aff_weights = model.afferent_weights[random_sample, 0] #- model.afferent_weights[random_sample, 1]
    aff_weights[0,0] = 0
    plt.subplot(4, 4, 2)
    plt.imshow(aff_weights.detach().cpu())
    plt.title(titles[1])

    # Afferent weights of a random sample
    net_afferent = model.current_afferent[0,0].detach().cpu() - model.thresholds[0,0].detach().cpu()
    net_afferent_bar = net_afferent + 0
    net_afferent_bar[0,0] = 0
    plt.subplot(4, 4, 3)
    plt.imshow(net_afferent_bar)
    plt.title(titles[2])

    # Lateral correlations of the random sample
    plt.subplot(4, 4, 4)
    plotvar = - model.inhibition[0, 0] + model.excitation[0,0]
    plotvar[0,0] = 0
    plt.imshow(plotvar.detach().cpu())
    plt.title(titles[3])

    # Lateral weights excitation of the random sample
    plt.subplot(4, 4, 5)
    plotvar = torch.relu(model.lateral_correlations[random_sample, 0] - 1.5/model.sheet_size**2)
    plt.imshow(plotvar.detach().cpu())
    plt.title(titles[4])

    # Model's current response
    plt.subplot(4, 4, 6)
    plt.imshow(model.current_response[0, 0].detach().cpu())
    plt.title(titles[5])

    # Histogram of the current response
    plt.subplot(4, 4, 7)
    hist = model.current_response.flatten().detach().cpu().numpy()
    plt.hist(hist[hist > 0])
    plt.title(titles[6])

    # Generate and display orientation and phase maps
    weights = model.afferent_weights.clone()
    M = int(np.sqrt(model.afferent_weights.shape[0]))  # Assuming MxM grid for reshaping
    ori_map, phase_map = get_orientations(weights, gabor_size=model.rf_size)
    ori_map = ori_map.reshape(M, M).cpu()
    phase_map = phase_map.reshape(M, M).cpu()
    
    # Orientation map
    plt.subplot(4, 4, 8)
    plt.imshow(ori_map, cmap='hsv')
    plt.title(titles[7])

    # Orientation histogram
    plt.subplot(4, 4, 9)
    plt.hist(ori_map.flatten())
    plt.title(titles[8])

    # Phase map
    plt.subplot(4, 4, 10)
    plt.imshow(phase_map, cmap='hsv')
    plt.title(titles[9])

    # Retinotopic Bias
    plt.subplot(4, 4, 11)
    l4_afferent = model.l4_afferent[0,0].cpu() - model.l4_thresholds[0,0].cpu()
    l4_afferent[0,0] = 0
    plt.imshow(l4_afferent)
    plt.title(titles[10])

    gabors = get_detectors(model.rf_size, 2, device='cuda')
    # Gabor detectors
    plt.subplot(4, 4, 12)
    plt.imshow(gabors[0,0].cpu())
    plt.title(titles[11])

    reco_input = network['activ'](network['model'](model.current_response))[0,0].detach().cpu()
    # nn reconstruction
    plt.subplot(4, 4, 13)
    plt.imshow(reco_input)
    plt.title(titles[12])

    # afferent with thresholds
    curr_net_afferent = torch.relu(net_afferent)
    curr_net_afferent[0,0] = 0
    plt.subplot(4, 4, 14)
    plt.imshow(curr_net_afferent.cpu())
    plt.title(titles[13])

    # thresholds
    thresholds = model.thresholds[0,0]
    #thresholds[0,0] = 1
    plt.subplot(4, 4, 15)
    plt.imshow(thresholds.cpu())
    plt.title(titles[14])

    # thresholds
    plt.subplot(4, 4, 16)
    plt.imshow(model.l4_response[0,0].cpu())
    plt.title(titles[15])

    print('Net Afferent Max: {:.3f}, Net Afferent Min: {:.3f}'. format(net_afferent.max(), net_afferent.min()))
    print('Mean Act Max: {:.3f}, Mean Act Min: {:.3f}'. format(model.mean_activations.max(), model.mean_activations.min()))
    print('Thresholds Max: {:.3f}, Thresholds Min: {:.3f}'. format(model.thresholds.max(), model.thresholds.min()))
    print('L4 Thresholds Max: {:.3f}, L4 Thresholds Min: {:.3f}'. format(model.l4_thresholds.max(), model.l4_thresholds.min()))
    print('Mean thresholds: {:.3f} and mean OFF strength {:.3f}'.format(model.thresholds.mean(), model.off_strengths))
    print('Mean pos afferent Mean: {:.3f} and aff strength {:.3f}'.format(model.mean_pos_afferent.mean(), model.aff_strength))
    print('Mean current response: {:.3f}'.format(model.current_response.mean()))
    print('L4 Strength: {:.3f} strength: {:.3f} aff strength: {:.3f}'.format(model.l4_strength, model.strength, model.aff_strength))


    plt.show()