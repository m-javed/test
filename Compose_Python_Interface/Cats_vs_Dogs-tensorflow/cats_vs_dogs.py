import cv2      #need to install opencv-python library [pip install opencv-python]
import numpy as np
import os
import time
from random import shuffle
from tqdm import tqdm       #[pip install tqdm] for showing progress bar
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

time_start = time.time() #record program start time

#-----------------------------------------------------Control Switches--------------------------------------------------
CREATE_DATA = False     # True = create data from images, False = Load already saved data
SIMPLE_MODEL = False    # True = use simple DNN with few layers, False = use improved DNN with more layers
PLOT_IMAGES = True      # True = plot test images with their predicted labels, False = no plotting
LOAD_MODEL = True       # True = load already trained model, False = train and save model
#-----------------------------------------------------------------------------------------------------------------------

TRAIN_DIR = 'train'
TEST_DIR = 'test'
IMG_SIZE = 50
LR = 1e-3 #Learning Rate

MODEL_NAME = 'dogs-vs-cats-convnet'

def create_label(image_name):
    """ Create an one-hot encoded vector from image name """
    word_label = image_name.split('.')[-3]
    if word_label == 'cat':
        return np.array([1,0])
    elif word_label == 'dog':
        return np.array([0,1])


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR, img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img_data), create_label(img)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img_data), img_num])

    #shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

if CREATE_DATA:
    train_data = create_train_data()
    test_data = create_test_data()
else:
    train_data = np.load('train_data.npy')
    test_data = np.load('test_data.npy')

train = train_data[:-500]
test = train_data[-500:]

X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = [i[1] for i in train]

X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = [i[1] for i in test]

#----------------------------------Try the simpel model with few hidden layers------------------------------------------
if SIMPLE_MODEL:
    tf.reset_default_graph()

    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)

    model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10,
              validation_set=({'input': X_test}, {'targets': y_test}),
              snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    #-----------------------------Try Improved Model----------------------------------------------------------
else:
    tf.reset_default_graph()

    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)

#-----------------------------------fit model only once and save the model for later use--------------------------------
if not LOAD_MODEL:
    print('Fitting model to the training data...')
    model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10,
               validation_set=({'input': X_test}, {'targets': y_test}),
               snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    print('Fitting done!')
    #-------------------------------------------------------------------------------------------------------------------
    print('Saving trained model...')
    model.save('my_model')
    print('Saving Model done!')
else:
    #-----------------------------------Load the model if laready fitted------------------------------------------------
    model.load('my_model')

#-------------------------------------Save predicted labels to text file------------------------------------------------
print('Saving predicted labels to a text file...')
predicted_labels = ''
for num, data in enumerate(test_data):# replace test_data with test_data[:n] for only first n samples

    img_num = data[1]
    img_data = data[0]

    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 1:
        str_label = 'Dog'
    else:
        str_label = 'Cat'
    #print(str_label)
    predicted_labels+= "%s\t"%img_num
    predicted_labels+= "%s\n"%str_label
with open('predicted_labels.txt', 'w') as file:
    file.write(predicted_labels)
    print('Saving predicted labels done!')
#-----------------------------------------------------------------------------------------------------------------------

#----------------------------------------------Print Elapsed Time-------------------------------------------------------
time_end = time.time()      #record program end time
time_elapsed = time_end-time_start
m, s = divmod(time_elapsed, 60)
h, m = divmod(m, 60)
print ("Elapsed time:- %d:%02d:%02d (h:m:s) " % (h, m, s))
#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------Plot some test images-----------------------------------------------------
if PLOT_IMAGES:
    print('Plotting images...')
    fig = plt.figure(figsize=(16, 12))

    for num, data in enumerate(test_data[:16]):

        img_num = data[1]
        img_data = data[0]

        y = fig.add_subplot(4, 4, num + 1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
        model_out = model.predict([data])[0]

        if np.argmax(model_out) == 1:
            str_label = 'Dog'
        else:
            str_label = 'Cat'

        y.imshow(orig, cmap='gray')
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    print('Plotting done!\nPS: Close the plot window for the program to continue...')
    plt.show()      #Show returns when the last plot window is closed
    #Generallly speaking the call to show() is always the last statement in the program
#-----------------------------------------------------------------------------------------------------------------------