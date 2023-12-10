import os
import numpy as np
from PIL import Image
from glob import glob

# Path to the directory containing your images
source_directory = 'processed/eegData_images/'

# Create a subdirectory 'eegData_images' within the source directory
subdirectory = os.path.join(source_directory, 'eegData_images')
os.makedirs(subdirectory, exist_ok=True)

# Loop over all image files in the source directory
for image_path in glob(os.path.join(source_directory, '*.png')):  # Adjust the extension if needed
    # Open the image
    with Image.open(image_path) as img:
        # Convert the image to a numpy array
        image_array = np.array(img)

        # Generate the path for the .npy file
        base_name = os.path.basename(image_path)
        new_name = os.path.splitext(base_name)[0] + '.npy'
        npy_path = os.path.join(subdirectory, new_name)

        # Save the numpy array as a .npy file
        np.save(npy_path, image_array)
