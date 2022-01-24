import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
Online sources were used as research to create this model

References:
Title: KNN (K Nearest Neighbours) in Python
Author: Python Engineer
Availability: https://www.youtube.com/watch?v=ngLyX54e1LU&t=1130s
Date: 3 Sept 2019

Title: K Nearest Neighbors Application - Practical Machine Learning Tutorial with Python p.14
Author: sentdex
Availability: https://www.youtube.com/watch?v=1i0zu9jHN6U&ab_channel=sentdex
Date: 1 May 2016

Title: Creating Our K Nearest Neighbors Algorithm - Practical Machine Learning with Python p.16
Author: sentdex
Availability: https://www.youtube.com/watch?v=1i0zu9jHN6U&ab_channel=sentdex
Date: 5 May 2016

Title: K Nearest Neighbour from Scratch - Machine Learning Python
Author: Aladdin Persson
Availability: https://www.youtube.com/watch?v=QzAaRuDskyc&t=559s
Date: 24 Apr 2020

"""

class KNN:

    def __init__(self, k=5, distance='euclidean', minkowski_p_val=2):
        """
        Constructor for KNN Object
        :param k: Number of nearest neighbours
        :param distance: distance measure used
        :param minkowski_p_val: value for minkowski distance measure
        """
        self.k = k
        self.distance = {
            'euclidean': lambda x1, x2: np.sqrt((np.sum((x1 - x2) ** 2, axis=0))),
            'manhattan': lambda x1, x2: (np.abs(x1 - x2).sum()),
            'minkowski': lambda x1, x2: (np.abs(x1 - x2) ** self.minkowski_p_val).sum(axis=0) ** (
                    1 / self.minkowski_p_val)
        }[distance]
        self.minkowski_p_val = minkowski_p_val

    def fit(self, X, y):
        """
        Knn is lazy training model, just stores values
        :param X: Input variables
        :param y: output variable, the classification
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        creates predictions of class for all values in X_test
        :param X:  Input variables, the X_test values
        :return: the predicted classifications for the test data in an array
        """
        predicted_labels = [self.predict_one(x) for x in X]
        return np.array(predicted_labels)

    def predict_one(self, x):
        """
        helper method to return target class for a datapoint
        :param x: single sample of input variables
        :return: label for a single sample
        """
        # compute the distance
        distances = [(self.distance(x, x_train)) for x_train in self.X_train]
        # get nearest k samples labels
        k_idx = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_idx]
        # get the majority vote most common
        return self.majority_vote(k_labels)


    def majority_vote(self, list):
        """
        helper method to return most common value of k labels(mode) list
        :param list: labels of k nearest samples
        :return:
        """
        ones = 0
        zeros = 0
        for i in list:
            if i == 1:
                ones += 1
            if i == 0:
                zeros += 1
        if zeros > ones:
            label = 0
        if ones > zeros:
            label = 1
        return label


def accuracy(y_true, y_pred):
    """
    function that calculates accuracy score
    :param y_true: y_test values
    :param y_pred: predicted y values
    :return: accuracy score
    """
    return np.sum((y_true == y_pred) / len(y_true))


def main():
    #data preprocessing including standardization, data split, balancing classes
    data = pd.read_csv("final_data.csv", index_col=[0])
    X = data.iloc[:, :-1]
    Y = data['winner']
    X = X.to_numpy()
    Y = Y.to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(X), Y, test_size=0.2)

    #creating model, fit and predicting values, printing accuracy
    knn = KNN(k=73, distance='manhattan')
    knn.fit(x_train, y_train)
    predictions = knn.predict(x_test)
    print("KNN model accuracy performance: " + str(accuracy(y_test, predictions)))


if __name__ == '__main__':
    main()
