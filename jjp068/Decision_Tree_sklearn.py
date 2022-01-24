import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
import sklearn.metrics as metrics
from sklearn.tree import plot_tree
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
model1 = DecisionTreeClassifier()
model1 = model1.fit(x_train,y_train)
prediction = model1.predict(x_test)
print("Model 1 performance metrics:")
print(metrics.classification_report(y_test,prediction))
accuracy = metrics.accuracy_score(y_test,prediction)
print("Accuracy for 'out of box' DT classifier: "+ str(accuracy))

#parameter search space, list indicates range
parameters_random = {'max_depth': [1, 600],
                     'splitter': ['best', 'random'],
                     'min_samples_leaf': [1, 1500],
                     'min_samples_split': [2, 200],
                     'criterion': ['gini', 'entropy'],
                     'ccp_alpha': [0, 100]
                     }
# hyperparameter optimization with RandomizedSearch CV function
rand_grid = RandomizedSearchCV(DecisionTreeClassifier(), parameters_random, cv=5, scoring='accuracy', n_iter=10,
                               random_state=42)
rand_grid.fit(x_train, y_train)
print("Best accuracy score across 5 fold cross validation training: "+str(rand_grid.best_score_))
print(rand_grid.best_params_)
print("Best estimator with parameters: ")
print(rand_grid.best_estimator_)

#creation of optimized model and classification report, accuracy score
model2 = rand_grid.best_estimator_
model2.fit(x_train, y_train)
prediction = model2.predict(x_test)
print("Model 2 performance metrics:")
print(metrics.classification_report(y_test,prediction))
print("Accuracy for optimised DT classifier: "+ str(metrics.accuracy_score(y_test,prediction)))

# plot for confusion matrix and feature importance
plot_confusion_matrix(model2,x_test,y_test)
plt.show()
ft_imprtnces = pd.Series(model2.feature_importances_, index= X.columns)
ft_imprtnces = ft_imprtnces.nlargest(10).plot(kind= 'bar')
plt.title('Top 10 Important Features')
plt.tight_layout()
plt.show()


#plot for tree visualization
"""
fig = plt.figure(figsize= (15,10))
tree_visual = plot_tree(model2, feature_names= X.columns, class_names= ['Fighter 2', 'Fighter1'], filled=True)
plt.tight_layout()
plt.savefig('DecisionTree.pdf')
plt.show()
"""

# statistic hypothesis test of "out of the box" model and optimised model
t,p = paired_ttest_5x2cv(model1,model2,X=X,y=y, scoring='accuracy', random_seed= 1)
print('P-value: %.3f, t-Statistic: %.3f' % (p, t))

if p<= 0.05:
    print("We reject null hypothesis")
else:
    print("We do not reject null hypothesis")
