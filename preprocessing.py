import numpy as np
import os
import glob
from PIL import Image

# Specify the path to your dataset images and masks
image_dir = "/research/d5/gds/hzyang22/data/ESD_seg/01/image"
mask_dir = "/research/d5/gds/hzyang22/data/ESD_seg/01/mask"

# Create a directory to store the CSV file
if not os.path.exists('data'):
    os.makedirs('data')

# Load the dataset images
images = []
image_files = glob.glob(os.path.join(image_dir, "*.png")) 
for image_file in image_files:
    image = np.array(Image.open(image_file))
    images.append(image)

# Load the dataset masks
masks = []
mask_files = glob.glob(os.path.join(mask_dir, "*.png")) 
for mask_file in mask_files:
    mask = np.array(Image.open(mask_file))
    masks.append(mask)

# Convert the images and masks to NumPy arrays
images_array = np.array(images)
masks_array = np.array(masks)

# Get the number of samples
num_samples = images_array.shape[0]

# Reshape the image and mask arrays to have a single row
images_flat = images_array.reshape(num_samples, -1)
masks_flat = masks_array.reshape(num_samples, -1)

# Create an array for the indices
indices = np.arange(num_samples).reshape(num_samples, 1)

# Concatenate the indices, images, and masks along the column axis
combined_array = np.concatenate((indices, images_flat, masks_flat), axis=1)

# Save the combined array as CSV file
np.savetxt('data/ESD.csv', combined_array, delimiter=',', fmt='%d')
