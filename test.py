import numpy as np
import cv2
import pickle
import sys

### Constants
FRAME_WIDTH = 640 # Set width of the video frame
FRAME_HEIGHT = 480 # Set height of the video frame
BRIGHTNESS = 180 # Set brightness level
THRESHOLD = 0.90 # Set probability threshold for predictions
IMG_DIM_RES = (32, 32) # resize to 32x32 pixels
FONT = cv2.FONT_HERSHEY_SIMPLEX # Set the font for displaying text

### Functions
def grayscale(img):
	"""Convert the input image to grayscale."""
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return img

def equalize(img):
	"""Apply histogram equalization to the input image to enhance contrast."""
	img = cv2.equalizeHist(img)
	return img

def preprocessing(img):
	"""Preprocess the input image by converting to grayscale and equalizing it."""
	img = grayscale(img)
	img = equalize(img)
	img = img / 255 # normalize pixel values to the range [0, 1] instead of [0, 255]
	return img

def getCalssName(classNo):
	"""Return the name of the traffic sign based on the class index"""
	if classNo == 0: return 'Speed Limit 20 km/h'
	elif classNo == 1: return 'Speed Limit 30 km/h'
	elif classNo == 2: return 'Speed Limit 50 km/h'
	elif classNo == 3: return 'Speed Limit 60 km/h'
	elif classNo == 4: return 'Speed Limit 70 km/h'
	elif classNo == 5: return 'Speed Limit 80 km/h'
	elif classNo == 6: return 'Speed Limit 100 km/h'
	elif classNo == 7: return 'Speed Limit 120 km/h'
	elif classNo == 8: return 'No passing'
	elif classNo == 9: return 'Right-of-way at the next intersection'
	elif classNo == 10: return 'Priority road'
	elif classNo == 11: return 'Yield'
	elif classNo == 12: return 'Stop'
	elif classNo == 13: return 'No vehicles'
	elif classNo == 14: return 'Vehicles over 3.5 metric tons prohibited'
	elif classNo == 15: return 'No entry'
	elif classNo == 16: return 'General caution'
	elif classNo == 17: return 'Dangerous curve to the left'
	elif classNo == 18: return 'Dangerous curve to the right'
	elif classNo == 19: return 'Double curve'
	elif classNo == 20: return 'Bumpy road'
	elif classNo == 21: return 'Slippery road'
	elif classNo == 22: return 'Road narrows on the right'
	elif classNo == 23: return 'Road work'
	elif classNo == 24: return 'Pedestrians'
	elif classNo == 25: return 'Children crossing'
	elif classNo == 26: return 'Bicycles crossing'
	elif classNo == 27: return 'Beware of ice/snow'
	elif classNo == 28: return 'Wild animals crossing'
	elif classNo == 29: return 'End of all speed and passing limits'
	elif classNo == 30: return 'Turn right ahead'
	elif classNo == 31: return 'Turn left ahead'
	elif classNo == 32: return 'Ahead only'
	elif classNo == 33: return 'Go straight or right'
	elif classNo == 34: return 'Go straight or left'
	elif classNo == 35: return 'Keep right'
	elif classNo == 36: return 'Keep left'
	elif classNo == 37: return 'Roundabout mandatory'



### Main
def main():
	# Initialize video capture from the default camera (index 0)
	video_cap = cv2.VideoCapture(0)
	video_cap.set(3, FRAME_WIDTH) # Set the width of the frame
	video_cap.set(4, FRAME_HEIGHT) # Set the height of the frame
	video_cap.set(10, BRIGHTNESS) # Set the brightness of the frame

	# take path from command line argument
	trained_model_path = sys.argv[1]
	while not trained_model_path:
		trained_model_path = input("Enter the path to the trained model: ")

	# Load the trained model using pickle
	pickle_in = open(trained_model_path, "rb")
	model = pickle.load(pickle_in)
	last_class = "" # Store the last class index

	while True:
		# Capture a frame from the camera
		success, original_frame = video_cap.read()
		flipped_frame = cv2.flip(original_frame, 1)
		
		# Convert the captured frame to a numpy array and resize it
		img = np.asarray(original_frame)
		img = cv2.resize(img, IMG_DIM_RES) # Resize to 32x32 pixels

		# Preprocess the image
		img = preprocessing(img)
		cv2.imshow("Processed Image", img) # Show the processed image
		
		# Reshape the image to match the input shape of the model
		img = img.reshape(1, 32, 32, 1)
		
		# Display placeholder text for classification and probability on the original image
		cv2.putText(flipped_frame, "CLASS: ", (20, 35), FONT, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
		cv2.putText(flipped_frame, "PROBABILITY: ", (20, 75), FONT, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
		cv2.putText(flipped_frame, "LAST: ", (20, 115), FONT, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

		# Make predictions using the model
		predictions = model.predict(img)
		class_id = np.argmax(predictions, axis=1) # Get the index of the class with the highest probability
		probability_value = np.amax(predictions) # Get the maximum probability value

		# Check if the probability exceeds the defined threshold
		if probability_value > THRESHOLD:
			class_name = getCalssName(class_id)
			print(class_name) # Print the class name based on the index
			# Display the class name and probability on the original image
			cv2.putText(flipped_frame, str(class_id) + " " + str(class_name), (120, 35), FONT, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
			cv2.putText(flipped_frame, str(round(probability_value * 100, 2)) + "%", (180, 75), FONT, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
			last_class = str(class_id) + " " + str(class_name)

		cv2.putText(flipped_frame, str(last_class), (100, 115), FONT, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
		
		# Show the result image with the class and probability
		cv2.imshow("Result", flipped_frame)
		
		# Exit the loop when 'q' is pressed
		if cv2.waitKey(1) and 0xFF == ord('q'):
			break

	# Release the camera and close all OpenCV windows
	video_cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()