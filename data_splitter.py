import os
import re
import shutil

# Define the path to the train data directory
train_data_dir = './dataset/train_data'
val_data_dir = './dataset/val_data'
test_data_dir = './dataset/test_data'

def extract_number(filename):
    """ Extracts the last number from a filename """
    # Use regex to find the last number in the filename
    match = re.search(r'_(\d+)(?=\.\w+)', filename)
    return int(match.group(1)) if match else float('inf')  # Return inf if no match found

# Loop through each class folder in the train directory
for class_dir in os.listdir(train_data_dir):
    class_path = os.path.join(train_data_dir, class_dir)
    
    if os.path.isdir(class_path):
        # Create class-specific directories in val_data and test_data
        val_class_dir = os.path.join(val_data_dir, class_dir)
        test_class_dir = os.path.join(test_data_dir, class_dir)
        
        os.makedirs(val_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        
        # Get a list of all image files in the class directory
        images = [img for img in os.listdir(class_path) if img.endswith(('jpg', 'jpeg', 'png'))]

        # Sort the files based on the extracted number
        images = sorted(images, key=extract_number)

        # Move the first 10 images to the validation folder
        for img in images[:10]:
            src = os.path.join(class_path, img)
            dest = os.path.join(val_class_dir, img)
            shutil.move(src, dest)
            print(f'Moved {src} to {dest}')

        # Move the next 10 images to the test folder
        for img in images[-10:]:  # Moving the next 10 images
            src = os.path.join(class_path, img)
            dest = os.path.join(test_class_dir, img)
            if os.path.exists(src):  # Check if the file still exists
                shutil.move(src, dest)
                print(f'Moved {src} to {dest}')

print("Splitting of images into validation and test folders completed.")
