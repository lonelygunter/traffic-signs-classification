import numpy as np
import cv2
import pickle
import sys
import torch
import pandas
import time

### Constants
FRAME_WIDTH = 640 # Set width of the video frame
FRAME_HEIGHT = 480 # Set height of the video frame
BRIGHTNESS = 180 # Set brightness level
THRESHOLD = 0.95 # Set probability threshold for predictions
IMG_DIM_RES = (32, 32) # resize to 32x32 pixels
FONT = cv2.FONT_HERSHEY_SIMPLEX # Set the font for displaying text

### Functions
def grayscale(img):
	import torchvision.transforms as transforms
	from torchvision.transforms import functional
	"""function to convert the image to grayscale"""
	if img is None or not isinstance(img, np.ndarray) or len(img.shape) == 2:  # Assuming 'grayscale' starts here
		return img  # return it as is
	else:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
		return img

def equalize(img):
	"""Function to equalize the histogram for contrast adjustment."""
	if img is None:
		return img  # Return None as is

	# Check if img is a tensor and convert to np array
	if isinstance(img, torch.Tensor):
		img = img.detach().cpu().np()  # Convert to np array

	if not isinstance(img, np.ndarray):
		print("Received non-array image: ", type(img))
		return img

	img = img.astype(np.uint8)  # Ensure the image is 8-bit grayscale
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
	if len(sys.argv) < 3:
		print(f"syntax: python3 {sys.argv[0]} <trained_model_path> <labels_path>")
		sys.exit(1)

	# create log file
	logfile = open("traffic_sign.log", "a")

	# load labels
	model_path = sys.argv[1]

	# load labels
	labels = pandas.read_csv(sys.argv[2])

	# Initialize video capture from the default camera (index 0)
	video_cap = cv2.VideoCapture(0)
	video_cap.set(3, FRAME_WIDTH) # Set the width of the frame
	video_cap.set(4, FRAME_HEIGHT) # Set the height of the frame
	video_cap.set(10, BRIGHTNESS) # Set the brightness of the frame

	# Load the model
	pickle_in = open(model_path, "rb")
	model = pickle.load(pickle_in)

	last_class = "" # Store the last class index

	while True:
		# Capture a frame from the camera
		success, original_frame = video_cap.read()
		flipped_frame = cv2.flip(original_frame, 1)

		# Convert the captured frame to a np array and resize it
		img = np.asarray(original_frame)
		img = cv2.resize(img, IMG_DIM_RES) # Resize to 32x32 pixels

		# Preprocess the image
		img = preprocessing(img)
		cv2.imshow("Processed Image", img) # Show the processed image

		# Reshape the image to match the input shape of the model
		img = img.reshape(1, 32, 32, 1)

		# Display placeholder text for classification and probability on the original image
		# cv2.putText(flipped_frame, "CLASS: ", (20, 35), FONT, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
		# cv2.putText(flipped_frame, "PROBABILITY: ", (20, 75), FONT, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
		cv2.putText(flipped_frame, "LAST: ", (20, 115), FONT, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
		# cv2.putText(flipped_frame, "PROB: ", (20, 155), FONT, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

		# Make predictions using the model
		predictions = np.array(model.predict(img))
		# normalize it
		predictions = predictions / np.linalg.norm(predictions)
		# class_id = model.predict_classes(img)
		class_id = np.argmax(predictions, axis=1) # Get the index of the class with the highest probability
		probability_value = np.amax(predictions) - 0.015 # Get the maximum probability value
		# cv2.putText(flipped_frame, str(probability_value) + "%", (120, 155), FONT, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

		# Check if the probability exceeds the defined threshold
		if probability_value > THRESHOLD:
			class_name = labels.loc[labels['ClassId'] == int(class_id), 'Name'].values[0]
			# Display the class name and probability on the original image
			# cv2.putText(flipped_frame, str(class_id) + " " + str(class_name), (120, 35), FONT, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
			# cv2.putText(flipped_frame, str(round(probability_value * 100, 2)) + "%", (180, 75), FONT, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
			last_class = str(class_id) + " " + str(class_name)
			logfile.write(f"{str(class_id)} {str(class_name)} w prob {probability_value}\n")


		cv2.putText(flipped_frame, str(last_class), (120, 115), FONT, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
		
		# Show the result image with the class and probability
		cv2.imshow("Result", flipped_frame)

		# Exit the loop when 'q' is pressed
		if cv2.waitKey(1) and 0xFF == ord('q'):
			break

	# Release the camera and close all OpenCV windows
	video_cap.release()
	cv2.destroyAllWindows()

	# close log file
	logfile.close()

if __name__ == "__main__":
	main()
