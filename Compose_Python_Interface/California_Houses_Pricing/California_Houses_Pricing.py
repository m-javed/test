#Estimating house value
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split as TTS
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.datasets.california_housing import fetch_california_housing

cal_hs = fetch_california_housing()
# split 80/20 train-test
X_train, X_test, y_train, y_test = TTS(cal_hs.data,
													cal_hs.target,
													test_size=0.2,
													random_state=1)
names = cal_hs.feature_names
print("Training GBRT...")
clf = GBR(n_estimators=100, max_depth=4,
								learning_rate=0.1, loss='huber',
								random_state=1)
clf.fit(X_train, y_train)
print(" done.")
accuracy = clf.score(X_test,y_test)
f_i = clf.feature_importances_
y_predict = clf.predict(X_test)



