from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import RandomizedSearchCV
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

Title: XGBoost
Author: XGBoost team
Availability: https://github.com/dmlc/xgboost
Version :  1.4.2

"""

#data import, data pre-processing including random undersampling, splitting data and standardizing
data = pd.read_csv("final_data.csv", index_col=[0])

X = data.iloc[:, :-1]
y = data['winner']

rus = RandomUnderSampler()
X, y = rus.fit_resample(X, y)
x_train, x_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(X), y, test_size=0.2)

#creation of out the box model and classification report, accuracy score
model1 = XGBClassifier()
model1 = model1.fit(x_train, y_train)
prediction = model1.predict(x_test)
print("Model 1 performance metrics:")
print(metrics.classification_report(y_test,prediction))
accuracy = metrics.accuracy_score(y_test, prediction)
print("Accuracy for 'out of box' XGBoost classifier: " + str(accuracy))

#parameter search space, list indicates range
param_grid = {
    'max_depth': [4, 6],
    'eta': [0.05, 0.1],
    'reg_lambda': [0, 10.0],
    'gamma': [0, 1],
    'scale_pos_weight': [1],
}

# hyperparameter optimization with RandomizedSearch CV function
rand_grid = RandomizedSearchCV(XGBClassifier(use_label_encoder=False), param_grid, cv=10, scoring='accuracy', n_iter=10,
                               random_state=42)
rand_grid.fit(x_train, y_train)
print("Best accuracy score across 5 fold cross validation training: " + str(rand_grid.best_score_))
print(rand_grid.best_params_)
print("Best estimator and params: ")
print(rand_grid.best_estimator_)

#creation of optimized model and classification report, accuracy score
model2 = rand_grid.best_estimator_
prediction = model2.predict(x_test)
print("Metric for model 2:")
print(metrics.classification_report(y_test, prediction))
print("Accuracy for optimised XGBoost classifier: " + str(metrics.accuracy_score(y_test, prediction)))

# plot for confusion matrix and ft importances
plot_confusion_matrix(rand_grid, x_test, y_test)
plt.show()
ft_imprtnces = pd.Series(model2.feature_importances_, index=X.columns)
ft_imprtnces = ft_imprtnces.nlargest(10).plot(kind='bar')
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


