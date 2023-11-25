import os
import numpy as np
from PIL import Image

def find_csv_files(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        if 'cleaned' in subdir or 'augmented' in subdir:
            for file in files:
                if file.endswith('.csv'):
                    yield os.path.join(subdir, file), 'augmented' if 'augmented' in subdir else 'cleaned'

def convert_to_npy(csv_path, npy_dir, index, subdir_type):
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    npy_path = os.path.join(npy_dir, f"{subdir_type}_{index}.npy")
    np.save(npy_path, data.T)
    return npy_path

def copy_and_rename_image(csv_path, img_dir, index, subdir_type):
    parts = csv_path.split(os.sep)
    csv_base_name = parts[-1].replace('_cleaned.csv', '').replace('_eyes_closed', '').replace('_augmented.csv', '')

    base_path = os.path.join('raw', parts[2], 'images', csv_base_name + '.png')
    if (parts[2] != 'experiment1'):
        base_path = base_path.replace('_recording', '')
    
    if os.path.exists(base_path):
        img = Image.open(base_path)
        new_size = tuple([int(dim * 0.66) for dim in img.size])
        resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        new_image_path = os.path.join(img_dir, f"{subdir_type}_{index}.png")
        resized_img.save(new_image_path)
        
        return new_image_path
    else:
        print(f"Image not found: {base_path}")
        return None

def main():
    processed_dir = 'processed/csv_raw/'
    npy_dir = 'processed/eegData_npy/'
    img_dir = 'processed/eegData_images/'

    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    index = 1
    for csv_file, subdir_type in find_csv_files(processed_dir):
        npy_path = convert_to_npy(csv_file, npy_dir, index, subdir_type)
        copy_and_rename_image(csv_file, img_dir, index, subdir_type)
        print(f"Processed {csv_file} into {npy_path}")
        index += 1

if __name__ == '__main__':
    main()
