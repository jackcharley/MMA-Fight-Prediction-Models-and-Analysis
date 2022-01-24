import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import plot_confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from mlxtend.evaluate import paired_ttest_5x2cv

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

Title: MLXtend
Author: mlxtend team
Availability: https://github.com/rasbt/mlxtend
Version:  0.18.0

Title: imblearn
Author: imblearn team
Availability: https://github.com/scikit-learn-contrib/imbalanced-learn
Version:  0.0

"""

#data import, data pre-processing including random undersampling, splitting data and standardizing
data = pd.read_csv("final_data.csv", index_col=[0])
X = data.iloc[:, :-1]
y = data['winner']
rus = RandomUnderSampler()
X, y = rus.fit_resample(X, y)
x_train, x_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(X), y, test_size=0.2)

#creation of out the box model and classification report, accuracy score
model1 = KNeighborsClassifier()
model1 = model1.fit(x_train,y_train)
prediction = model1.predict(x_test)
print("Model 1 performance metrics:")
print(metrics.classification_report(y_test,prediction))
accuracy = metrics.accuracy_score(y_test,prediction)
print("Accuracy for 'out of box' KNeighbours classifier: "+ str(accuracy))

#parameter search space, list indicates range
params = {'n_neighbors': [3, 255], 'weights': ['uniform', 'distance'],
          'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'p': [1, 2],
          'metric': ['euclidean', 'minkowski', 'chebyshev', ]}

# hyperparameter optimization with RandomizedSearch CV function
rand_grid = RandomizedSearchCV(KNeighborsClassifier(), params, cv=5, scoring='accuracy', n_iter=10, random_state=42)
rand_grid.fit(x_train, y_train)
print("Best accuracy score across 5 fold cross validation training: "+ str(rand_grid.best_score_))
print(rand_grid.best_params_)
print("Best estimator and parameters: ")
print(rand_grid.best_estimator_)

#creation of optimized model and classification report, accuracy score
model2 = rand_grid.best_estimator_
y_true, y_predict = (y_test, model2.predict(x_test))
print(metrics.classification_report(y_true,y_predict))
print("Accuracy for optimised KNeighbours classifier: "+str(metrics.accuracy_score(y_true, y_predict)))

# plot for confusion matrix
plot_confusion_matrix(rand_grid, x_test, y_test)
plt.show()

# statistic hypothesis test of "out of the box" model and optimised model
t,p = paired_ttest_5x2cv(model1,model2,X=X,y=y, scoring='accuracy', random_seed= 1)
print('P-value: %.3f, t-Statistic: %.3f' % (p, t))

if p<= 0.05:
    print("We reject null hypothesis")
else:
    print("We do not reject null hypothesis")