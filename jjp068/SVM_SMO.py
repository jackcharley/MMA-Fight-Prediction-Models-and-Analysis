import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import cross_val_score
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

"""
The code in this class was adapted from the paper "Yet More Simple SMO algorithm" by Tymchyshyn and Khavliuk

Title: simplest_smo_ever/simple_svm.ipynb
Author: Tymchyshyn and Khavliuk
Date: 25/02/21
Code Version:1
Availability: https://github.com/fbeilstein/simplest_smo_ever/blob/master/simple_svm.ipynb

"""

class SVM:
    def __init__(self, kernel='linear', C=10, max_iter=100, degree=3, gamma=1):
        """
        :param kernel: Type of kernel used to manipulate decision boundary
        :param C: Inverse strength of L2 regularization
        :param max_iter: Number of iterations to change and update decision function
        :param degree: degree of the power (used in polynomial function)
        :param gamma: the influence of training examples by distance (low values far away, high values close by)
        """
        self.kernel = {'poly': lambda x, y: np.dot(x, y.T) ** degree,
                       'rbf': lambda x, y: np.exp(-gamma * np.sum((y - x[:, np.newaxis]) ** 2, axis=-1)),
                       'linear': lambda x, y: np.dot(x, y.T)}[kernel]
        self.C = C
        self.max_iter = max_iter

    def restrict_to_square(self, t, v0, u):
        """
        restricts value to a square around the objective function so that it does not violate: 0<ai<C

        :param t:
        :param v0: value for the constraint. Transpose of the idxM and idxL
        :param u:
        :return:
        """
        t = (np.clip(v0 + t * u, 0, self.C) - v0)[1] / u[1]
        return (np.clip(v0 + t * u, 0, self.C) - v0)[0] / u[0]

    def fit(self, X, y):
        """
        :param X: Input variables
        :param y: output variable classification

        """
        self.X = X.copy()
        #converts values of y to -1,1 as is the convention in SVM
        self.y = y * 2 - 1
        #initializes lagrangian multipliers to 0
        self.lambdas = np.zeros_like(self.y, dtype=float)
        #objective function
        self.K = self.kernel(self.X, self.X) * self.y[:, np.newaxis] * self.y

        #for loop iterating for iterations in constructor
        for _ in range(self.max_iter):
            #for loop for every lagrangrian multiplier
            for idxM in range(len(self.lambdas)):
                # sets second multiplier, a random choice of all lagrangian multipliers
                idxL = np.random.randint(0, len(self.lambdas))
                # matrix of all possible 2 value combination of the two lagrangian multipliers
                Q = self.K[[[idxM, idxM], [idxL, idxL]], [[idxM, idxL], [idxM, idxL]]]
                # value for the constraint. Transpose of the idxM and idxL
                v0 = self.lambdas[[idxM, idxL]]
                # # scalar function for a sample, minimizing the objective function with idxM, idxL
                k0 = 1 - np.sum(self.lambdas * self.K[[idxM, idxL]], axis=1)
                # scalar function for a sample, minimizing the objective function with idxM, idxL
                u = np.array([-self.y[idxL], self.y[idxM]])
                # maintaining lambda values as positive
                t_max = np.dot(k0, u) / (np.dot(np.dot(Q, u), u) + 1E-15)
                # lambda values after restriction
                self.lambdas[[idxM, idxL]] = v0 + u * self.restrict_to_square(t_max, v0, u)
        # indexes of support vectors
        idx, = np.nonzero(self.lambdas > 1E-15)
        #bias (mean error, taken from classifications
        self.b = np.mean((1.0 - np.sum(self.K[idx] * self.lambdas, axis=1)) * self.y[idx])

    def decision_function(self, X):
        """
        :param X: X input values
        :return: decision function, boundary for classification
        """
        return np.sum(self.kernel(X, self.X) * self.y * self.lambdas, axis=1) + self.b

    def predict(self, X):
        return (np.sign(self.decision_function(X)) + 1) // 2

def accuracy(y_true, y_pred):
        return np.sum((y_true == y_pred) / len(y_true))

def main():
    # data preprocessing including standardization, data split, balancing classes
    data = pd.read_csv("final_data.csv", index_col=[0])
    data = data.to_numpy()# save memory
    data = np.float16(data)
    X = data[:, :-1]
    Y = data[:, -1:]
    rus = RandomUnderSampler()
    X, y = rus.fit_resample(X, Y)
    X_train, X_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(X), y, test_size=0.2)

    # creating model, fit and predicting values, printing accuracy
    clf = SVM(C=10,kernel= 'rbf', max_iter = 100, gamma= 0.01)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_true = [0 if y == -1 else y for y in y_pred]
    print(accuracy(y_test,y_pred_true))

if __name__=='__main__':
    main()




