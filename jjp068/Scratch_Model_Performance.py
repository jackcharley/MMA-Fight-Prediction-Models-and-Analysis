from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from Statistical_Analysis_Models import plot_results
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import ttest_rel

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

"""

#data preprocessing including standardization, data split, balancing classes
data = pd.read_csv("final_data.csv", index_col=[0])
X = data.iloc[:, :-1]
y = data['winner']
rus = RandomUnderSampler()
X, y = rus.fit_resample(X, y)
ss = StandardScaler()
X = ss.fit_transform(X)

# accuracy scores taken from the 3 models "from scratch"
log_reg_scores = [0.6627565982404691,0.6715542521994133, 0.6412512218963831,
                  0.6568914956011729,0.6725317693059627, 0.6695992179863146,
                0.6930596285434993, 0.649071358748778, 0.6529814271749754, 0.6695992179863146
                  ]
knn_scores = [
0.6880466472303207,0.6861030126336248,0.7142857142857142, 0.6812439261418853,
0.6812439261418853,0.6754130223517978,0.6812439261418853, 0.6831875607385811,
0.6754130223517978,0.6686103012633625
]
svm_scores = [
0.7233626588465296, 0.7331378299120233, 0.7253176930596283,
0.7194525904203322, 0.745845552297165, 0.7096774193548385,
0.7321603128054739, 0.7429130009775169,0.7262952101661777,
0.7106549364613879
]

# dictionary of scikit learn models

api_models = {
    'knn': KNeighborsClassifier(algorithm='brute', metric='euclidean', n_neighbors=255,
                                p=1, weights='distance'),
    'log_reg': LogisticRegression(C=0.1, tol=0.001, solver='sag', penalty='l2',
                                  max_iter=3500),
    'svm': SVC(C=1.0, break_ties= False, cache_size = 200, class_weight= None, coef0= 0.0, decision_function_shape = 'ovr', degree= 3, gamma= 'scale')
}

#compiling scores for scratch models
scores1 = log_reg_scores + knn_scores + svm_scores

# getting scores for scikit learn models
scores2 = []
for model in api_models.keys():
    score2 = cross_val_score(api_models.get(model), X, y, cv=10, scoring='accuracy')
    scores2.append(score2)
#flattening array for scores for 3rd party models
scores2 = np.concatenate(scores2)

#printing scores
print('Original models, Mean, St.dev'+ str(np.mean(scores1))+", "+str(np.std(scores1)))
print('Sk-learn models, Mean, St.dev'+ str(np.mean(scores2))+", "+str(np.std(scores2)))

#statistically testing sample
t_stat, p_val2 = ttest_rel(scores1, scores2)
print("paired t-test, comparing means of the original models and methods from API")
print("Test statistic: " + str(t_stat) + " P value: " + str(p_val2))
if p_val2 <= 0.05:
    print('We reject null hypothesis')
else:
    print('We do not reject the null hypothesis')

#boxplot of results
plot_results([scores1,scores2], ['Originals','Sk-learn'])

