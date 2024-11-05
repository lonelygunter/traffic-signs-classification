import os
import re
import shutil
import sys

# Define the path to the train data directory
VAL_PROP = 0.1
TEST_PROP = 0.1

def extract_number(filename):
    """ Extracts the last number from a filename """
    # Use regex to find the last number in the filename
    match = re.search(r'_(\d+)(?=\.\w+)', filename)
    return int(match.group(1)) if match else float('inf')  # Return inf if no match found

# setup paths
train_path = sys.argv[1] + "/train_data"
val_path = sys.argv[1] + "/val_data"
test_path = sys.argv[1] + "test_data"

# Loop through each class folder in the train directory
for class_dir in os.listdir(train_path):
    class_path = os.path.join(train_path, class_dir)
    
    if os.path.isdir(class_path):
        # Create class-specific directories in val_data and test_data
        val_class_dir = os.path.join(val_path, class_dir)
        test_class_dir = os.path.join(test_path, class_dir)
        
        os.makedirs(val_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        
        # Get a list of all image files in the class directory
        images = [img for img in os.listdir(class_path) if img.endswith(('jpg', 'jpeg', 'png'))]

        # Get number of images in the train directory
        num_images = len(images)

        # Calculate the number of images to move to the validation and test directories
        num_val_images = int(num_images * VAL_PROP)
        num_test_images = int(num_images * TEST_PROP)

        # Sort the files based on the extracted number
        images = sorted(images, key=extract_number)

        # Move the first 10 images to the validation folder
        for img in images[:num_val_images]:
            src = os.path.join(class_path, img)
            dest = os.path.join(val_class_dir, img)
            shutil.move(src, dest)
            print(f'Moved {src} to {dest}')

        # Move the next 10 images to the test folder
        for img in images[-num_test_images:]:  # Moving the next 10 images
            src = os.path.join(class_path, img)
            dest = os.path.join(test_class_dir, img)
            if os.path.exists(src):  # Check if the file still exists
                shutil.move(src, dest)
                print(f'Moved {src} to {dest}')

print("Splitting of images into validation and test folders completed.")
