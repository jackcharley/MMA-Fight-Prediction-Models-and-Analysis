import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import sklearn.metrics as metrics
from sklearn.model_selection import RandomizedSearchCV
from mlxtend.evaluate import paired_ttest_5x2cv
from tensorflow.keras.models import load_model

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

Title: TensorFlow
Author: TensorFlow Team
Availability: https://github.com/tensorflow/tensorflow
Version: 2.6.0

"""

#data import, data pre-processing including random undersampling, splitting data and standardizing
data = pd.read_csv("final_data.csv", index_col = [0])
X = data.iloc[:,:-1]
y= data['winner']
rus = RandomUnderSampler(random_state=42)
X, y = rus.fit_resample(X,y)
X= StandardScaler().fit_transform(X)
#encoder = load_model('encoder.h5', compile=False)
#X = encoder.predict(X)
x_train, x_test, y_train , y_test = train_test_split(X, y)

#creation of out the box model and classification report, accuracy score
model1 = MLPClassifier()
model1 = model1.fit(x_train,y_train)
prediction = model1.predict(x_test)
print("Model 1 performance metrics:")
print(metrics.classification_report(y_test,prediction))
accuracy = metrics.accuracy_score(y_test,prediction)
print("Accuracy for 'out of box' MLP NN: "+ str(accuracy))

#parameter search space, list indicates range
params ={
    'hidden_layer_sizes':[30,300],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver':['lbfgs', 'sgd', 'adam'],
    'alpha':[0.0001, 0.001],
    'learning_rate':['constant', 'invscaling', 'adaptive'],
    'tol':[0.0001,0.1],
    'max_iter':[200,1000],
    'n_iter_no_change': [10,100]
}

# hyperparameter optimization with RandomizedSearch CV function
rand_grid = RandomizedSearchCV(MLPClassifier(), params, cv= 5, scoring = 'accuracy', n_iter= 10, random_state= 42)
rand_grid.fit(x_train,y_train)
print("Best accuracy score across 5 fold cross validation training: "+str(rand_grid.best_score_))
print("Best estimator and parameters: ")
print(rand_grid.best_estimator_)
print(rand_grid.best_params_)

#creation of optimized model and classification report, accuracy score
model2 = rand_grid.best_estimator_
prediction = model2.predict(x_test)
print("Model 2 metrics: ")
print(metrics.classification_report(y_test,prediction))
print("Accuracy for optimised MLP: " + str(metrics.accuracy_score(y_test,prediction)))

plot_confusion_matrix(model2,x_test,y_test)
plt.show()

# plot for confusion matrix
t,p = paired_ttest_5x2cv(model1,model2,X=X,y=y, scoring='accuracy', random_seed= 1)
print('P-value: %.3f, t-Statistic: %.3f' % (p, t))

if p<= 0.05:
    print("We reject null hypothesis")
else:
    print("We do not reject null hypothesis")

