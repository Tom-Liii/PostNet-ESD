def get():
    # Set the path to the dataset folder
    dataset_path = Path('/research/d5/gds/hzyang22/data/new_esd_seg')

    # Initialize the lists
    train_data_file_names = []
    val_data_file_names = []
    test_data_file_names = []

    # Iterate over the subdirectories
    for subdir in os.listdir(dataset_path):
        sub_dir_path = os.path.join(dataset_path, subdir)
        if os.path.isdir(sub_dir_path):
            # Iterate over the image files in the current subdirectory
            image_files = []
            for file in os.listdir(sub_dir_path):
                if file.endswith(".png"):
                    image_files.append(os.path.join(sub_dir_path, file))
            
            # Shuffle the image files
            random.shuffle(image_files)
            
            # Split the image files into three parts: train, val, and test
            num_files = len(image_files)
            train_split = int(0.7 * num_files)  # 70% for training
            val_split = int(0.2 * num_files)   # 20% for validation
            train_data_file_names.extend(image_files[:train_split])
            val_data_file_names.extend(image_files[train_split:train_split+val_split])
            test_data_file_names.extend(image_files[train_split+val_split:])

    # Print the resulting lists
    print("Train data files:")
    for file_name in train_data_file_names:
        print(file_name)

    print("\nValidation data files:")
    for file_name in val_data_file_names:
        print(file_name)

    print("\nTest data files:")
    for file_name in test_data_file_names:
        print(file_name)
    return train_data_file_names, val_data_file_names, test_data_file_names
