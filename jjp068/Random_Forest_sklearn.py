import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import sklearn.metrics as metrics
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
data = pd.read_csv("final_data.csv", index_col = [0])
X = data.iloc[:,:-1]
y= data['winner']
x_train, x_test, y_train , y_test = train_test_split(StandardScaler().fit_transform(X), y)

#creation of out the box model and classification report, accuracy score
model1 = RandomForestClassifier()
model1 = model1.fit(x_train,y_train)
prediction = model1.predict(x_test)
accuracy = metrics.accuracy_score(y_test,prediction)
print("Model 1 performance metrics:")
print(metrics.classification_report(y_test,prediction))
print("Accuracy for 'out of box' Random forest classifier: "+ str(accuracy))

#parameter search space, list indicates range
params = {'n_estimators': [200,300], 'criterion':['gini','entropy'], 'max_depth': [1,10],
          'min_samples_split': [2,10], 'min_samples_leaf':[10,100], 'max_features': ['auto',None]}

# hyperparameter optimization with RandomizedSearch CV function
rand_grid = RandomizedSearchCV(RandomForestClassifier(), params, cv = 5, scoring ='accuracy', n_iter=10, random_state=42)
rand_grid.fit(x_train, y_train)
print("Best accuracy score across 5 fold cross validation training: "+str(rand_grid.best_score_))
print(rand_grid.best_params_)
print ("Best estimator parameters: ")
print(rand_grid.best_estimator_)

#creation of optimized model and classification report, accuracy score
model2 = rand_grid.best_estimator_
model2.fit(x_train,y_train)
y_true , y_predict = (y_test, model2.predict(x_test))
print("Model 2 performance metrics:")
print(metrics.classification_report(y_true,y_predict))
new_accuracy = metrics.accuracy_score(y_test, y_predict)
print("Accuracy for optimised Random forest classifier: "+ str(new_accuracy))

# plot for confusion matrix and ft importances
plot_confusion_matrix(rand_grid, x_test, y_test)
plt.show()

ft_imprtnces = pd.Series(model2.feature_importances_, index= X.columns)
ft_imprtnces = ft_imprtnces.nlargest(10).plot(kind= 'bar')
plt.title('Top 10 Important Features')
plt.tight_layout()
plt.show()

# statistic hypothesis test of "out of the box" model and optimised model
t,p = paired_ttest_5x2cv(model1,model2,X=X,y=y, scoring='accuracy', random_seed= 1)
print('P-value: %.3f, t-Statistic: %.3f' % (p, t))

if p<= 0.05:
    print("We reject null hypothesis")
else:
    print("We do not reject null hypothesis")