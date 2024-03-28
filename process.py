import os
import numpy as np
import cv2
from scipy.ndimage import gaussian_laplace
from skimage import exposure, img_as_ubyte
from skimage.io import imread, imsave
from tqdm import tqdm

import cv2
from scipy.ndimage import gaussian_laplace
from skimage import exposure, img_as_ubyte
from skimage.io import imread

import numpy as np
import cv2
from scipy.ndimage import gaussian_laplace, gaussian_filter
from skimage.io import imread, imsave
from skimage import img_as_ubyte

def process_image(image_path):
    # Read the image
    img = imread(image_path)
    
    # Convert to grayscale if necessary
    if img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
    
    gray_img = np.array(gray_img) / 255.
    # Apply Laplacian of Gaussian edge detection
    log_img = gaussian_laplace(gray_img, sigma=2.5)
    
    # Threshold and normalize
    threshold = 0
    log_img[log_img < threshold] = 0
    log_img = log_img / log_img.max()
    
    # Define Gaussian kernel size and standard deviation
    sigma = 4  # Standard deviation for Gaussian kernel
    kernel_size = int(6*sigma + 1)  # Ensure kernel size covers enough area
    
    # Apply Gaussian filter to compute the neighborhood average
    neighborhood_avg = gaussian_filter(log_img, sigma=sigma, mode='constant')
    
    # Define epsilon to avoid division by zero
    eps = 1e-1
    saturation = 1
    
    # Apply Contrast Gain Control
    cgc_img = log_img / (neighborhood_avg + eps)
    
    # Normalize the CGC image to have values in the range [0, 1]
    cgc_img = np.tanh(cgc_img*saturation)
    
    # Convert to 8-bit format
    processed_img_8bit = img_as_ubyte(cgc_img)
    
    # Downsize the image by a factor of 2 in each dimension
    height, width = processed_img_8bit.shape[:2]
    downsized_img = cv2.resize(processed_img_8bit, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    
    return downsized_img

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
        
        # Save the processed image
        output_path = os.path.join(output_folder, os.path.basename(file_path))
        imsave(output_path, processed_img)

# Example usage
root_folder = './imagenet_mini_raw'  # Change this to your folder path
output_folder = './imagenet_mini'  # Change this to your desired output folder path
process_images_in_folder(root_folder, output_folder)
