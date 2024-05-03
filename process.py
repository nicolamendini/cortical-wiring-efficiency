import os
import numpy as np
import cv2
from skimage import exposure, img_as_ubyte
from skimage.io import imread, imsave
from tqdm import tqdm
from scipy.ndimage import gaussian_laplace, gaussian_filter
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

def process_image(image_path):

    # Read the image
    img = imread(image_path)
    
    # Convert to grayscale if necessary
    if img.ndim >= 3:
        gray_img = rgb2gray(img[:,:,:3])  # using skimage's rgb2gray for simplicity
    else:
        gray_img = img
    
    gray_img = np.array(gray_img, dtype=float)
    
    # Apply Laplacian of Gaussian edge detection
    log_img = gaussian_laplace(gray_img, sigma=1.2)
    inv_log_img = -log_img  # Inverse polarity for the OFF channel
    
    # Normalize the ON and OFF channels
    on_channel = normalize_channel(log_img)   
    off_channel = normalize_channel(inv_log_img)
    
    # Create a dummy third channel (black)
    dummy_channel = np.zeros_like(on_channel)
    
    # Stack channels to form a 3-channel image
    processed_img = np.stack((on_channel, off_channel, dummy_channel), axis=-1)
    
    return processed_img
    
# Normalize each channel
def normalize_channel(channel):
	threshold = 1e-4
	channel[channel < threshold] = 0
	channel = channel / np.max(channel) if np.max(channel) != 0 else channel

	# Gaussian filter parameters
	sigma = 4
	kernel_size = int(6 * sigma + 1)

	# Local normalization
	neighborhood_avg = gaussian_filter(channel, sigma=sigma, mode='constant')
	eps = 3e-3
	saturation = 0.45
	cgc_img = np.tanh((channel / (neighborhood_avg + eps)) * saturation)

	# Convert to 8-bit
	return img_as_ubyte(cgc_img)

def process_images_in_folder(root_folder, output_folder):
    # Get all image paths first to calculate progress
    image_paths = []
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                image_paths.append(os.path.join(subdir, file))
                
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process images with progress bar
    for file_path in tqdm(image_paths, desc="Processing images"):
        processed_img = process_image(file_path)
        
        # Change file extension to '.png' for the output
        output_filename = os.path.splitext(os.path.basename(file_path))[0] + '.png'
        output_path = os.path.join(output_folder, output_filename)
        
        # Save the processed image in PNG format
        imsave(output_path, processed_img, format='png')

# Example usage
root_folder = './imagenet-mini'  # Change this to your folder path
output_folder = './input_stimuli'  # Change this to your desired output folder path
process_images_in_folder(root_folder, output_folder)



