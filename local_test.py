import numpy as np
import cv2
import pickle
import sys
import torch
import numpy
import os
import pandas
from glob import glob
from tqdm import tqdm

### Constants
THRESHOLD = 0.90 # Set probability threshold for predictions
IMG_DIM_RES = (32, 32) # resize to 32x32 pixels
IMG_DIM = (32,32,3) # dimensions of the images (height, width, channels)

### Functions
def get_all_imgs(folder_path):
	# Define common image file extensions
	image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp')

	# Initialize an empty list to store image paths
	image_paths = []

	# Loop through each extension type to find image files
	for ext in image_extensions:
		# Use glob to find all images with the given extension in the folder
		images = glob(os.path.join(folder_path, '**', ext), recursive=True)
		image_paths.extend(images)

	return image_paths

def grayscale(img):
	import torchvision.transforms as transforms
	from torchvision.transforms import functional
	"""function to convert the image to grayscale"""
	if img is None or not isinstance(img, numpy.ndarray) or len(img.shape) == 2:  # Assuming 'grayscale' starts here
		return img  # return it as is
	else:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
		return img

def equalize(img):
	"""Function to equalize the histogram for contrast adjustment."""
	if img is None:
		return img  # Return None as is

	# Check if img is a tensor and convert to NumPy array
	if isinstance(img, torch.Tensor):
		img = img.detach().cpu().numpy()  # Convert to NumPy array

	if not isinstance(img, numpy.ndarray):
		print("Received non-array image: ", type(img))
		return img

	img = img.astype(numpy.uint8)  # Ensure the image is 8-bit grayscale
	img = cv2.equalizeHist(img)  # Apply histogram equalization

	return img

def preprocessing(img):
	"""Preprocess the input image by converting to grayscale and equalizing it."""
	img = grayscale(img)
	img = equalize(img)
	img = img / 255 # normalize pixel values to the range [0, 1] instead of [0, 255]
	return img

### Main
def main():
	# take path from command line argument
	if len(sys.argv) < 2:
		print(f"syntax: python3 {sys.argv[0]} <trained_model_path> <labels_path> <imgs_dir>")
		sys.exit(1)

	# Load the model
	pickle_in = open(sys.argv[1], "rb")
	model = pickle.load(pickle_in)

	# load labels
	labels = pandas.read_csv(sys.argv[2])

	# imgs directory
	img_dir = sys.argv[3]

	if ".DS_Store" in os.listdir(img_dir):
		os.remove(os.path.join(img_dir, ".DS_Store"))

	input_imgs = get_all_imgs(img_dir)
	good_predictions = 0

	for input_img in (pbar := tqdm(input_imgs)):
		img = cv2.imread(input_img)
		img = cv2.resize(img, IMG_DIM_RES) # Resize to 32x32 pixels

		# Preprocess the image
		img = preprocessing(img)

		# Reshape the image to match the input shape of the model
		img = img.reshape(1, 32, 32, 1)

		# Make predictions using the model
		predictions = np.array(model.predict(img, verbose=0))
		# print(predictions)
		# class_id = model.predict_classes(img)
		class_id = np.argmax(predictions, axis=1) # Get the index of the class with the highest probability
		probability_value = np.amax(predictions) # Get the maximum probability value
		pbar.set_description(f"acc:{good_predictions/len((input_imgs)):.2f}")

		if probability_value > THRESHOLD and int(class_id) == int(os.path.basename(os.path.dirname(input_img))):
			class_name = labels.loc[labels['ClassId'] == int(class_id), 'Name'].values[0]

			# Display the class name and probability on the original image
			# print("-----------")
			# print(input_img)
			# print(f"PROB: {probability_value} %")
			# print(f"CLASS: {class_id} {class_name}")
			good_predictions += 1

	print(f"good predictions: {good_predictions}/{len((input_imgs))}")
	print(f"accuracy:{good_predictions/len((input_imgs))}")



if __name__ == "__main__":
	main()