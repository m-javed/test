**Cats vs. Dogs**
=============================================
In this project we train a neural network algorithm to distinguish dogs from cats. Two models have been trained, first one with few hidden layers and the second one with more hidden layers. Accuracy of first model was quite low, ~50%, almost same as tossing a coin. Second model shows an improved accuracy of almost 85%.
  
Dataset Description
-------------------------------------------

 The training archive contains 25,000 images of dogs and cats. Train your algorithm on these files and predict the labels for test1.zip (1 = dog, 0 = cat).

Code Description
-----------------------
If you are running the program for the first time, you need to convert images to numbers to create the dataset and save the dataset to disk. Later, when you have already created the dataset, you just need to load data from disk. This is achieved by following lines of code
 `# If dataset is not created
 #train_data = create_train_data()
 #test_data = create_test_data()
 #If you have already created the dataset:
 train_data = np.load('train_data.npy')
 test_data = np.load('test_data.npy')`

Please use the 'fit model only once and save the model for later use' section to train the model only if you can't load the saved model (named as 'my_model') or the training data has changed. Otherwise, it is recommended to load the trained model to save time.

If some machine doesn't have graphics support, please use the 'Save predicted labels to text file' section and manually compare the predicted labels with the corresponding images.

Example Predicted Labels
--------------------------
Here are some of the test images with their corresponding predicted labels.
![Image not found!](/images/predicted_labels.png)

> **Note:** You need to rename folders containing train images to **'train'** and test images to **'test'**
> - **Cats vs Dogs** dataset can be downloaded [here](https://www.kaggle.com/c/dogs-vs-cats)
> - You can find more information about **tensorflow** library [here](http://tensorflow.org/)
> - This program makes use of cv2 library which can be installed by running `pip install opencv-python` 
