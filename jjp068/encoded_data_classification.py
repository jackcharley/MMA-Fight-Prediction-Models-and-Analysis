import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import sklearn.metrics as metrics
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.models import load_model
from sklearn.neural_network import MLPClassifier

"""
References:

Title: Scikit-Learn
Author: Scikit-Learn Team
Availability: https://github.com/scikit-learn/scikit-learn
Version: 0.24.2

Title: matplotlib
Author: matplotlib Team
Availability: https://github.com/matplotlib/matplotlib
Version: 3.4.2

Title: numpy
Author: numpy Team
Availability: https://github.com/numpy/numpy
Version: 1.19.5

Title: pandas
Author: pandas Team
Availability: https://github.com/pandas-dev/pandas
Version:  1.2.4

Title: SciPy
Author: Scipy team
Availability: https://github.com/scipy/scipy
Version:  1.7.0

Title: imblearn
Author: imblearn team
Availability: https://github.com/scikit-learn-contrib/imbalanced-learn
Version:  0.0

Title: TensorFlow
Author: TensorFlow Team
Availability: https://github.com/tensorflow/tensorflow
Version: 2.6.0

"""

#data preprocessing steps, split data, random undersampling
data = pd.read_csv("final_data.csv", index_col=[0])
X = data.iloc[:, :-1]
y = data['winner']
rus = RandomUnderSampler()
X, y = rus.fit_resample(X, y)
normalized_X = preprocessing.normalize(X)
x_train, x_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(normalized_X), y, test_size=0.2)
# encoding dataset with trained encoder using importing function
encoder = load_model('encoder.h5', compile= False)
x_train_e =encoder.predict(x_train)
x_test_e = encoder.predict(x_test)
print(x_train.shape, x_train_e.shape)


#testing different models with encoded dataset
model = LogisticRegression(C=0.1, tol=0.001, solver='sag', penalty='l2',
                                                            max_iter=3500)
model.fit(x_train_e, y_train)
prediction = model.predict(x_test_e)
print("Logistic regression - accuracy: "+ str(metrics.accuracy_score(y_test,prediction)))

model2  = SVC(C=1, degree=1, tol=0.1, kernel='rbf', gamma='scale')
model2.fit(x_train_e,y_train)
prediction2 = model2.predict(x_test_e)
print("SVM - accuracy: " + str(metrics.accuracy_score(y_test,prediction2)))

model3 = MLPClassifier(activation='logistic', alpha=0.001, hidden_layer_sizes=100, tol=0.001)
model3.fit(x_train_e,y_train)
prediction3 = model3.predict(x_test_e)
print("Neural network - accuracy: "+ str(metrics.accuracy_score(y_test, prediction3)))

model4 = KNeighborsClassifier(algorithm='brute', metric='euclidean', n_neighbors=255,
                                              p=1, weights='distance')
model4.fit(x_train_e,y_train)
prediction4 = model4.predict(x_test_e)
print('K nearest neighbours'+str(metrics.accuracy_score(y_test, prediction4)))




