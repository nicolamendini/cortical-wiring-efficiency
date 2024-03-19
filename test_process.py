import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import random
import glob

def sample_and_plot_histogram(target_folder):
    # Find all image paths
    image_paths = glob.glob(os.path.join(target_folder, '*.*'))
    
    # Ensure there are enough images
    if len(image_paths) < 100:
        print("Not enough images in the folder. Found only", len(image_paths), "images.")
        return
    
    # Randomly sample 100 image paths
    sampled_paths = random.sample(image_paths, 100)
    
    # Initialize an array to accumulate histograms
    combined_histogram = np.zeros(256)
    
    # Compute and accumulate histograms
    for path in sampled_paths:
        img = imread(path)
        if len(img.shape) > 2:  # Convert to grayscale if the image is colored
            img = np.mean(img, axis=2).astype(np.uint8)
        # Calculate histogram for nonzero values only
        histogram, _ = np.histogram(img[img > 0], bins=np.arange(257))
        combined_histogram += histogram
    
    # Plot the combined histogram
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(256), combined_histogram, color='gray')
    plt.title('Histogram of Nonzero Pixel Values Across 100 Sampled Images')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 255])
    plt.show()

# Example usage
target_folder = './imagenet_mini'  # Change this to your target folder path
sample_and_plot_histogram(target_folder)
