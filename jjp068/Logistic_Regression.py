import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from imblearn.under_sampling import RandomUnderSampler

"""
Online sources were used as research to create this model

References:
Title: Logistic Regression in Python
Author: Python Engineer
Availability: https://www.youtube.com/watch?v=JDU3AzH3WKg&t=2s
Date: 15 Sept 2019

Title: Logistic Regression in Python from Scratch
Author: Coding Lane
Availability: https://www.youtube.com/watch?v=nzNp05AyBM8
Date: 4 Feb 2021

Title: Logistic Regression from Scratch - Machine Learning Python
Author: Aladdin Persson
Availability: https://www.youtube.com/watch?v=x1ez9vi611I
Date: 29 May 2020

"""

class logistic_regress:

    def __init__(self, alpha=0.01, penalty=1, C=10, iterations=10000):
        """
        :param alpha: learning rate, step for gradient descent
        :param penalty: if penalty non zero then L2 regularization is included
        :param C: Inverse strength of regularization
        :param iterations: how many steps model runs for
        """
        self.alpha = alpha
        self.penalty = penalty
        self.C = C
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        :param X: input variable samples
        :param y: output variable classifications
        """
        samples, features = X.shape

        # initialise weights ( can be zero or random)
        self.weights = np.zeros(features)
        self.bias = 0

        # iteratively update weights
        for i in range(self.iterations):
            # linear plotting of line function
            linear_model = np.dot(X, self.weights) + self.bias
            # transformation to probability using sigmoid function
            y_predicted = self.sigmoid(linear_model)

        if self.penalty == None:
            self.log_odds(X, y_predicted, y, samples)
        else:
            self.reg_log_odds(X, y_predicted, y, samples)

    def predict(self, x):
        """
        :param x: samples of x input variables
        :return: y output prediction for x sample
        """
        linear_model = np.dot(x, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        # list comprehension to classify into 1 and 0
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_class


    def sigmoid(self, x):
        """
        sigmoid function helper method, activation function
        :param x:
        :return:
        """
        return 1 / (1 + np.exp(-x))


    def log_odds(self, x, y_pred, y, samples):
        """
        # updating weights using derivatives

        :param x: samples of input variables
        :param y_pred: y predicted output
        :param y: y true output
        :param samples: X samples

        """
        # updating weights using derivatives
        # derivative of weights
        dw = (1 / samples) * np.dot(x.T, (y_pred - y))
        # derivative of bias
        db = (1 / samples) * np.sum(y_pred - y)

        #gradient descent, alpha defines learning rate
        self.weights -= self.alpha * dw
        self.bias -= self.alpha * db

    def reg_log_odds(self, x, y_pred, y, samples):
        """
        # updating weights using derivatives including L2 regularization

        :param x: sample of input variable
        :param y_pred: y predicted output
        :param y: y true output
        :param samples: X samples

        """
        # updating weights using derivatives including L2 regularization
        # derivative of weights
        dw = ((1 / samples) * (self.C * (np.dot(x.T, (y_pred - y)))) + np.sum(self.weights))
        # derivative of bias
        db = (1 / samples) * np.sum(y_pred - y)

        self.weights -= self.alpha * dw
        self.bias -= self.alpha * db


def accuracy(y_true, y_pred):
    """
    function that calculates accuracy score
    :param y_true: y_test values
    :param y_pred: predicted y values
    :return: accuracy score
    """
    return np.sum((y_true == y_pred) / len(y_true))


def main():
    #data importing and pre-processing
    data = pd.read_csv("final_data.csv", index_col=[0])
    X = data.iloc[:, :-1]
    Y = data['winner']
    rus = RandomUnderSampler()
    X, y = rus.fit_resample(X, Y)
    x_train, x_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(X), y, test_size=0.2)

    #fitting and training model
    log_reg_model = logistic_regress(alpha=0.001, iterations=1500, penalty=1, C= 9)
    log_reg_model.fit(x_train, y_train)
    predictions = log_reg_model.predict(x_test)
    print("Logistic regression accuracy performance: " + str(accuracy(y_test, predictions)))


if __name__ == '__main__':
    main()
