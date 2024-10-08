import os
import cv2
import numpy
import pandas
import matplotlib.pyplot as plt
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pickle

### Constants
DATA_PATH = "./media/myData" # path to the dataset
LABEL_PATH = './media/labels.csv' # path to the labels csv file
BATCH_SIZE_VAL = 50 # size of batches during training
STEPS_X_EPOCH = 2000 # number of steps per epoch
N_EPOCHS = 30 # number of epochs for training
IMG_DIM = (32,32,3) # dimensions of the images (height, width, channels)
TEST_DATA = 0.2 # proportion of the data to use for testing (1000 images split will 200)
VAL_DATA = 0.2 # proportion of training data to use for validation (1000 images 20% of remaining 800 will be 160 for validation)

### Functions
def grayscale(img):
    """Convert the input image to grayscale."""
    if len(img.shape) == 2: # if image is already grayscale
        return img # return it as is
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
        return img

def equalize(img):
    """Apply histogram equalization to the input image to enhance contrast."""
    img = img.astype(numpy.uint8) # ensure the image is 8-bit grayscale
    img = cv2.equalizeHist(img) # apply histogram equalization
    return img

def preprocessing(img):
    """Preprocess the input image."""
    img = grayscale(img) # convert to grayscale
    img = equalize(img) # equalize histogram
    img = img / 255 # normalize pixel values to the range [0, 1] instead of [0, 255]
    return img

def t_sign_model():
    """Return the model for traffic sign classification."""
    n_kernels = 60 # number of kernels
    kernel_size_5x5 = (5, 5) # size of the kernels (5x5)
    kernel_size_3x3 = (3, 3) # size of the second kernel (3x3)
    pool_size = (2, 2) # size of the pooling window (2x2): reduces img size by half to prevent overfitting
    n_nodes = 500 # number of nodes in the fully connected layer

    # initialize the model
    model = Sequential()

    # First convolutional block (add more conv layers = less features but more accurate)
    model.add((Conv2D(n_kernels, kernel_size_5x5, input_shape=(IMG_DIM[0], IMG_DIM[1], 1), activation='relu')))
    model.add((Conv2D(n_kernels, kernel_size_5x5, activation='relu'))) # 2nd conv layer
    model.add(MaxPooling2D(pool_size=pool_size)) # max pooling layer
    
    # Second convolutional block
    model.add((Conv2D(n_kernels // 2, kernel_size_3x3, activation='relu'))) # 3rd conv layer with smaller kernel size
    model.add((Conv2D(n_kernels // 2, kernel_size_3x3, activation='relu'))) # 4th conv layer
    model.add(MaxPooling2D(pool_size=pool_size)) # max pooling layer
    model.add(Dropout(0.5)) # dropout layer to prevent overfitting
    
    # Fully connected layers
    model.add(Flatten()) # flatten the output for the fully connected layer
    model.add(Dense(n_nodes, activation='relu')) # fully connected layer with ReLU activation
    model.add(Dropout(0.5)) # dropout layer for regularization
    model.add(Dense(n_classes, activation='softmax')) # output layer with softmax for multi-class classification

    # compile the model with Adam optimizer and categorical cross-entropy loss
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model



### Main

# importing of the Images
class_id = 0 # index for classes
images = [] # list for image data
class_labels = [] # list for class labels
classes = os.listdir(DATA_PATH) # list all folders (classes)
n_classes = len(classes) # total number of classes
print("Total Classes Detected:", n_classes)

# load the images
print("Importing Classes.....")
for x in range (0, n_classes):
    imgs = os.listdir(DATA_PATH+"/"+str(class_id)) # list images in class folder

    for y in imgs: # loop through each image
        cur_img = cv2.imread(DATA_PATH+"/"+str(class_id)+"/"+y) # read image
        images.append(cur_img) # add image to the list
        class_labels.append(class_id) # add class label
    print(class_id, end =" ") # print class index
    class_id += 1
print(" ")

images = numpy.array(images) # convert list of images to numpy array
class_labels = numpy.array(class_labels) # convert list of class labels to numpy array

# split Data
    # X_train = array of images to train
    # y_train = corresponding class IDs
    # X_test = array of images to test
    # y_test = corresponding class IDs

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, class_labels, test_size=TEST_DATA)
# further split training data into training and validation sets
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=VAL_DATA)


# print the shapes of the datasets
print("Data Shapes")
print("Train", X_train.shape, y_train.shape) # print training data shape
print("Validation", X_validation.shape, y_validation.shape) # print validation data shape
print("Test", X_test.shape, y_test.shape) # print test data shape

# check if the number of images matches the number of labels for each dataset
assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels in the training set"
assert(X_validation.shape[0] == y_validation.shape[0]), "The number of images is not equal to the number of labels in the validation set"
assert(X_test.shape[0] == y_test.shape[0]), "The number of images is not equal to the number of labels in the test set"

# check if the image dimensions are correct
assert(X_train.shape[1:] == (IMG_DIM)), "The dimensions of the training images are wrong"
assert(X_validation.shape[1:] == (IMG_DIM)), "The dimensions of the validation images are wrong"
assert(X_test.shape[1:] == (IMG_DIM)), "The dimensions of the test images are wrong"

# read the label data from csv file

labels = pandas.read_csv(LABEL_PATH) # read the labels csv file
print("data shape ", labels.shape, type(labels))

# display some sample images from each class

sample_x_class = [] # list for the number of samples per class
n_cols = 5 # number of columns for displaying images
fig, axs = plt.subplots(nrows=n_classes, ncols=n_cols, figsize=(5, 300)) # create subplots
fig.tight_layout() # adjust layout to prevent overlap

# loop through and display images
for i in range(n_cols):
    for j, row in labels.iterrows(): # iterate over rows in the labels
        x_selected = X_train[y_train == j] # select images from the current class
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap=plt.get_cmap("gray")) # display a random image
        axs[j][i].axis("off") # hide axis (for cleaner display)

        if i == 2:
            axs[j][i].set_title(str(j) + "-" + row["Name"]) # set title to class label
            sample_x_class.append(len(x_selected)) # add the number of samples for this class


# plot the distribution of training dataset
print(sample_x_class)
plt.figure(figsize=(12, 4))
plt.bar(range(0, n_classes), sample_x_class) # create a bar chart for sample distribution
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()



# preprocess training, validation, and test sets
X_train = numpy.array(list(map(preprocessing, X_train)))
X_validation = numpy.array(list(map(preprocessing, X_validation)))
X_test = numpy.array(list(map(preprocessing, X_test)))

# add a depth of 1 to the images

# reshape the data to add a depth of 1 (grayscale images have only one channel)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# augmentation of images to make the model more robust

# create an image data generator for augmenting the training data
img_gen = ImageDataGenerator(width_shift_range=0.1, # randomly shift images horizontally by 10%
                             height_shift_range=0.1, # randomly shift images vertically by 10%
                             zoom_range=0.2, # randomly zoom in or out
                             shear_range=0.1, # randomly shear the images
                             rotation_range=10) # randomly rotate images by up to 10 degrees
img_gen.fit(X_train) # fit the generator on the training data
batches = img_gen.flow(X_train, y_train, batch_size=20) # generate augmented images
X_batch, y_batch = next(batches) # get the next batch

# display some augmented images
fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()
for i in range(15):
    axs[i].imshow(X_batch[i].reshape(IMG_DIM[0], IMG_DIM[1])) # reshape to 32x32 and display
    axs[i].axis('off') # hide axis (for cleaner display)
plt.show()

# convert labels to one-hot encoded format
y_train = to_categorical(y_train, n_classes)
y_validation = to_categorical(y_validation, n_classes)
y_test = to_categorical(y_test, n_classes)



# build and train the model
model = t_sign_model()

history = model.fit(
    img_gen.flow(X_train, y_train, batch_size=BATCH_SIZE_VAL), # train the model using data augmentation
    steps_per_epoch=STEPS_X_EPOCH, # number of steps per epoch
    epochs=N_EPOCHS, # number of epochs
    validation_data=(X_validation, y_validation), # validation data
    shuffle=1 # shuffle data during training
)

# plot the training and validation loss
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')

# plot the training and validation accuracy
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.show()

# evaluate the model on the test set
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0]) # print test loss
print('Test Accuracy:', score[1]) # print test accuracy

# save the trained model
pickle_out = open("model_trained.p", "wb")
pickle.dump(model, pickle_out) # serialize and save the model
pickle_out.close()

# wait for a key press to close OpenCV windows
cv2.waitKey(0)