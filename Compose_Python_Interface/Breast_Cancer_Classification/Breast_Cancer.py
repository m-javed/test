#Breast Cancer dataset
import numpy as np
import pandas as pd
from numpy import mean
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
cancer = load_breast_cancer()
#print(cancer.DESCR) # Print the data set description
#cancer is a disctionary, with following keys
cancer.keys()

# Check the number of features of the breast cancer dataset
len(cancer['feature_names'])

#Create cancer dataframe
data = np.c_[cancer.data, cancer.target]
columns = np.append(cancer.feature_names, 'target')
cancerdf  = pd.DataFrame(data, columns=columns)

#Check class distribution. (i.e. how many instances of malignant (encoded 0) and how many benign (encoded 1)?)
counts = cancerdf.target.value_counts(ascending=True)
counts.index = "malignant benign".split()

#Split the DataFrame into X (the data) and y (the labels).
X = cancerdf[cancerdf.columns[:-1]]
y = cancerdf.target
#Split into test train sets
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X,
														y, 
														test_size = 0.25,
														random_state = 0)
#fit KNN
from sklearn.neighbors import KNeighborsClassifier as KNN
knn = KNN(n_neighbors=1)
knn.fit(X_train,y_train)
#--Using our knn classifier, we predict the class label using the mean value for each feature.--

#We use cancerdf.mean()[:-1].values.reshape(1, -1)
#	...which gets the mean value for each feature, ignores the
#	...target column, and reshapes the data from 1 dimension to 2
#	...(necessary for the precict method of KNeighborsClassifier).
means = cancerdf.mean()[:-1].values.reshape(1, -1)
#print('Predicted Class label using mean value of each feature: ',knn.predict(means))
accu = knn.score(X_test,y_test)
print('Testing accuracy: ', accu)
def accuracy_plot():
	# Find the training and testing accuracies by target value (i.e. malignant, benign)
	mal_train_X = X_train[y_train==0]
	mal_train_y = y_train[y_train==0]
	ben_train_X = X_train[y_train==1]
	ben_train_y = y_train[y_train==1]

	mal_test_X = X_test[y_test==0]
	mal_test_y = y_test[y_test==0]
	ben_test_X = X_test[y_test==1]
	ben_test_y = y_test[y_test==1]

	scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y), 
			  knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]


	#~ plt.figure()

	#~ # Plot the scores as a bar chart
	#~ bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])

	#~ # directly label the score onto the bars
	#~ for bar in bars:
		#~ height = bar.get_height() #hight represents score, will be used to adjust vertical position of the label
		#~ plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), 
					 #~ ha='center', color='w', fontsize=11)

	#~ # remove all the ticks (both axes), and tick labels on the Y axis
	#~ plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

	#~ # remove the frame of the chart
	#~ for spine in plt.gca().spines.values():
		#~ spine.set_visible(False)

	#~ plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);
	#~ plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)
	#~ plt.show()
	return scores
scores = accuracy_plot()
#~ print('\n--Training and Test Accuracy by target value--\n')
#~ print('Training Accuracy for Malignant cells: ',scores[0])
#~ print('Training Accuracy for Benign cells: ',scores[1])
#~ print('Test Accuracy for Malignant cells: ',scores[2])
#~ print('Test Accuracy for Benign cells: ',scores[3])