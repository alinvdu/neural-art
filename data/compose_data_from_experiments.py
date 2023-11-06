import os
import numpy as np
import shutil
from PIL import Image

def find_csv_files(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.csv') and 'cleaned' in subdir:
                yield os.path.join(subdir, file)

def convert_to_npy(csv_path, npy_dir, index):
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    npy_path = os.path.join(npy_dir, f"{index}.npy")
    np.save(npy_path, data.T)
    return npy_path

def copy_and_rename_image(csv_path, img_dir, index):
    # Extract parts of the CSV path
    parts = csv_path.split(os.sep)
    # Remove the '_cleaned.csv' and the suffix '_eyes_closed' from the CSV filename
    csv_base_name = parts[-1].replace('_cleaned.csv', '').replace('_eyes_closed', '')
    
    # Construct the base path for the image
    base_path = os.path.join('raw', parts[2], 'images', csv_base_name + '.png')
    
    if os.path.exists(base_path):
        # Open and resize the image
        img = Image.open(base_path)
        new_size = tuple([int(dim * 0.66) for dim in img.size])
        resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Save the resized image in the new path
        new_image_path = os.path.join(img_dir, f"{index}.png")
        resized_img.save(new_image_path)
        
        return new_image_path
    else:
        print(f"Image not found: {base_path}")
        return None


def main():
    processed_dir = 'processed/csv_raw/'
    npy_dir = 'processed/eegData_npy/'
    img_dir = 'processed/eegData_images/'
    
    # Create directories if they don't exist
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    index = 1  # Start index from 1
    
    for csv_file in find_csv_files(processed_dir):
        npy_path = convert_to_npy(csv_file, npy_dir, index)
        copy_and_rename_image(csv_file, img_dir, index)
        print(f"Processed {csv_file} into {npy_path}")
        index += 1

if __name__ == '__main__':
    main()
