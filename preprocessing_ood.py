import numpy as np
import os
import glob
from PIL import Image
from tqdm import tqdm

# specify the data directory ids that you want to use
ids = {'04', '05', '06'}

# Initialize a flag to check if it's the first directory
first_directory = True

# Iterate over the directory IDs
for directory_id in ids:
    # Construct the image and mask directories based on the directory ID
    image_dir = f"/research/d5/gds/hzyang22/data/ESD_seg/{directory_id}/image"
    mask_dir = f"/research/d5/gds/hzyang22/data/ESD_seg/{directory_id}/mask"

    # Print the data locations
    print("Image Directory:", image_dir)
    print("Mask Directory:", mask_dir)

    # Create a directory to store the CSV file
    if not os.path.exists('data'):
        os.makedirs('data')

    # Print the data locations
    print("Image Directory:", image_dir)
    print("Mask Directory:", mask_dir)

    # Load the dataset images
    print("Loading images...")
    images = []
    image_files = glob.glob(os.path.join(image_dir, "*.png")) 
    for image_file in tqdm(image_files):
        image = np.array(Image.open(image_file))
        images.append(image)

    # Load the dataset masks
    print("Loading masks...")
    masks = []
    mask_files = glob.glob(os.path.join(mask_dir, "*.png")) 
    for mask_file in tqdm(mask_files):
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

    # Create column headers as consecutive numbers
    num_features = images_flat.shape[1] + masks_flat.shape[1]
    headers = [""] + [str(i) for i in range(num_features)]

    # Concatenate the indices, images, and masks along the column axis
    combined_array = np.concatenate((indices, images_flat, masks_flat), axis=1)

    # Save the combined array as CSV file
    csv_file_path = 'data/ESD_ood.csv'
    print("Appending dataset to CSV:", csv_file_path)

    # Check if it's the first directory, if yes, write the headers along with the data
    if first_directory:
        with open(csv_file_path, 'wb') as file:
            np.savetxt(file, combined_array, delimiter=',', fmt='%d', header=",".join(headers), comments='')
        first_directory = False
    else:
        # Append the data to the existing CSV file
        with open(csv_file_path, 'ab') as file:
            np.savetxt(file, combined_array, delimiter=',', fmt='%d')

    print("Data appended to CSV.")