**Dogs vs. Cats**
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

> **Note:** You need to rename folders containing train images to **'train'** and test images to **'test'**
> -- **Dogs vs Cats** dataset can be downloaded [here](https://www.kaggle.com/c/dogs-vs-cats)
> -- You can find more information about **tensorflow** library [here](http://tensorflow.org/)
> -- This program makes use of cv2 library which can be installed by running `pip install opencv-python` 
