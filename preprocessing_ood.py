import numpy as np
import os
import glob
from PIL import Image
from tqdm import tqdm

def preprocessing_ood(id):
    # Specify the path to your dataset images and masks
    image_dir = f"/research/d5/gds/hzyang22/data/ESD_seg/{id}/image"
    mask_dir = f"/research/d5/gds/hzyang22/data/ESD_seg/{id}/mask"

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
    csv_file_path = f'data/ESD_{id}.csv'
    print("Saving dataset as CSV:", csv_file_path)

    # Create a progress bar for saving
    with tqdm(total=1, desc="Saving CSV", unit="file") as pbar:
        np.savetxt(csv_file_path, combined_array, delimiter=',', fmt='%d', header=",".join(headers), comments='')
        pbar.update(1)

    print("Dataset saved as CSV.")

preprocessing_ood('01')