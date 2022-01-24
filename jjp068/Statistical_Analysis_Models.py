import numpy as np
import pandas as pd
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from sklearn.tree import plot_tree
from scipy.stats import f_oneway
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind
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


def data_split_with_processing():
    """

    :return: preprocessed data as X input variables y output variable
    """
    data = pd.read_csv("C:\\Users\\jackc\\OneDrive\Desktop\\jjp068\\final_data.csv", index_col=[0])
    X = data.iloc[:, :-1]
    y = data['winner']
    rus = RandomUnderSampler()
    X, y = rus.fit_resample(X, y)
    X = StandardScaler().fit_transform(X)
    # encoder = load_model('encoder.h5', compile=False)
    # X = encoder.predict(X)
    return X, y


def data_split_no_processing():
    """

    :return: data without processing
    """
    data = pd.read_csv("final_data.csv", index_col=[0])
    X2 = data.iloc[:, :-1]
    y2 = data['winner']
    return X2, y2


def get_accuracy(X, y, model_dict, model):
    """

    :param X: x input variables
    :param y: y ouput classification
    :param model_dict: dictionary containing models
    :param model: current model to be used
    :return: 30 accuracy scores
    """
    if model in model_dict:
        scores = []
        for i in range(3):
            score = cross_val_score(model_dict.get(model), X, y, cv=10, scoring='accuracy')
            scores.append(score)
    return np.concatenate(scores)


def plot_results(scores, labels):
    """

    :param scores: accuracy scores for the models
    :param labels: model names
    :return: boxplot of results
    """
    plt.boxplot(scores, labels=labels, showmeans=True)
    plt.xlabel("Model")
    plt.ylabel("Mean Accuracy")
    plt.show()


def main():
    #dictionary of models
    model_dict = {'decision_tree': DecisionTreeClassifier(ccp_alpha=0, criterion='entropy', max_depth=600,
                                                          splitter='random'),
                  'logistic_regression': LogisticRegression(C=0.1, tol=0.001, solver='sag', penalty='l2',
                                                            max_iter=3500),
                  'knn': KNeighborsClassifier(algorithm='brute', metric='euclidean', n_neighbors=255,
                                              p=1, weights='distance'),
                  'svm': SVC(C=1, degree=1, tol=0.1, kernel='rbf', gamma='scale'),
                  'random_forest': RandomForestClassifier(criterion='entropy', max_depth=10, max_features=None,
                                                          min_samples_leaf=10, min_samples_split=10,
                                                          n_estimators=100),
                  'adaboost': AdaBoostClassifier(algorithm='SAMME', learning_rate=1, n_estimators=100),
                  'xgboost': XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                           colsample_bynode=1, colsample_bytree=1, eta=0.1, gamma=1,
                                           gpu_id=-1, importance_type='gain', interaction_constraints='',
                                           learning_rate=0.100000001, max_delta_step=0, max_depth=6,
                                           min_child_weight=1,
                                           n_estimators=100, n_jobs=8, num_parallel_tree=1, random_state=0,
                                           reg_alpha=0, reg_lambda=10.0, scale_pos_weight=1, subsample=1,
                                           tree_method='exact', use_label_encoder=False,
                                           validate_parameters=1, verbosity=None),
                  'mlp': MLPClassifier()}
    model_accuracy_samples = []
    X, y = data_split_with_processing()
    # for every model get multiple samples for accuracy
    for model in model_dict.keys():
        accuracy_scores = get_accuracy(X, y, model_dict, model=model)
        print(model + " Accuracy Mean:" + str(np.mean(accuracy_scores)) + " std: " + str(np.std(accuracy_scores)))
        model_accuracy_samples.append(accuracy_scores)
    labels = ['dt', 'lr', 'knn', 'svm', 'rf', 'ada', 'xgb', 'mlp']
    plot_results(model_accuracy_samples, labels)
    f_stat, p_val = f_oneway(
        model_accuracy_samples[0], model_accuracy_samples[1], model_accuracy_samples[2], model_accuracy_samples[3],
        model_accuracy_samples[4], model_accuracy_samples[5], model_accuracy_samples[6], model_accuracy_samples[7])
    print("ANOVA one-way test, comparing means of the models")
    print("Test statistic: " + str(f_stat) + " P value: " + str(p_val))
    if p_val <= 0.05:
        print('We reject null hypothesis')
    else:
        print('We do not reject the null hypothesis')

    """
    comparing the performance of traditional methods and ensemble learning/ neural nets
    """
    print(' comparing the performance of traditional methods and ensemble learning / neural nets')
    traditional_methods_means = np.concatenate(model_accuracy_samples[:4])
    ensemble_nets_means = np.concatenate(model_accuracy_samples[4:])
    print('Traditional methods, Mean and Standard deviation' + str(np.mean(traditional_methods_means)) + ", " + str(
        np.std(traditional_methods_means)) + " Ensemble methods mean and stdev"+ str(np.mean(ensemble_nets_means))+", "+ str(np.std(ensemble_nets_means)))
    plot_results([traditional_methods_means, ensemble_nets_means], ['Traditional methods', 'Ensemble Methods'])
    t_stat, p_val2 = ttest_rel(traditional_methods_means, ensemble_nets_means)
    print("paired t-test, comparing means of the traditional models and ensemble methods and nn's")
    print("Test statistic: " + str(t_stat) + " P value: " + str(p_val2))
    if p_val2 <= 0.05:
        print('We reject null hypothesis')
    else:
        print('We do not reject the null hypothesis')

"""
print(sum(accuracy_scores) / len(accuracy_scores))
plt.hist(accuracy_scores, density=True, bins=10)
plt.ylabel('Probability')
plt.xlabel('Accuracy')
plt.show()
"""

if __name__ == '__main__':
    main()
