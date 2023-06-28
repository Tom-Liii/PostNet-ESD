from PIL import Image
import numpy as np
import os

# Define the range of directory IDs
start_id = 4
end_id = 6  # Update the end ID based on your requirement

# Create a directory to store the CSV files
if not os.path.exists('data'):
    os.makedirs('data')

# Iterate over the directory IDs
for directory_id in range(start_id, end_id + 1):
    # Convert the directory ID to a zero-padded string
    directory_id_str = str(directory_id).zfill(2)

    # Construct the image and mask directories based on the directory ID
    image_dir = f"/research/d5/gds/hzyang22/data/ESD_seg/{directory_id_str}/image"
    mask_dir = f"/research/d5/gds/hzyang22/data/ESD_seg/{directory_id_str}/mask"

    # Print the data locations
    print("Image Directory:", image_dir)
    print("Mask Directory:", mask_dir)

    # Load the dataset images
    print("Loading images...")
    images = []
    image_files = glob.glob(os.path.join(image_dir, "*.png"))  # Assuming images are in PNG format
    for image_file in tqdm(image_files):
        image = np.array(Image.open(image_file))
        images.append(image)

    # Load the dataset masks
    print("Loading masks...")
    masks = []
    mask_files = glob.glob(os.path.join(mask_dir, "*.png"))  # Assuming masks are in PNG format
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

    # Convert the array elements to integers
    images_flat = images_flat.astype(int)
    masks_flat = masks_flat.astype(int)

    # Create an array for the indices
    indices = np.arange(num_samples).reshape(num_samples, 1)

    # Create column headers as consecutive numbers
    num_features = images_flat.shape[1] + masks_flat.shape[1]
    headers = ["index"] + [str(i) for i in range(num_features)]

    # Concatenate the indices, images, and masks along the column axis
    combined_array = np.concatenate((indices, images_flat, masks_flat), axis=1)

    # Save the combined array as a CSV file
    csv_file_path = f'data/dataset_{directory_id_str}.csv'
    print("Saving dataset as CSV:", csv_file_path)
    np.savetxt(csv_file_path, combined_array, delimiter=',', fmt='%d', header=",".join(headers), comments='')

    print("Dataset saved as CSV.")

print("All data saved in CSV.")
