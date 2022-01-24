import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import ttest_ind

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

def data_split_with_processing(data):
    X = data.iloc[:, :-1]
    y = data['winner']
    rus = RandomUnderSampler()
    X, y = rus.fit_resample(X, y)
    X = StandardScaler().fit_transform(X)
    # encoder = load_model('encoder.h5', compile=False)
    # X = encoder.predict(X)
    return X, y



def main():
    data = pd.read_csv("final_data.csv", index_col=[0])

    # getting information for fights with female competitors using weight class

    female_fights = (data[data['Weight class'] < 4])

    # female fighter categoricals
    female_fights_cat = female_fights[
        [ 'f1_Wins', 'f1_Losses', 'f1_DOB', 'Weight class', 'Round', 'f2_Wins', 'f2_Losses', 'f2_DOB',
         'Method_DEC','Method_KO/TKO','Method_SUB','f1_STANCE_Orthodox', 'f1_STANCE_Southpaw',
           'f1_STANCE_Switch', 'f1_STANCE_unk', 'f2_STANCE_Orthodox',
           'f2_STANCE_Southpaw', 'f2_STANCE_Switch', 'f2_STANCE_unk']]
    female_fights_num = female_fights[['f1_Height_cm','f1_SLpM', 'f1_Str. Acc.', 'f1_SApM',
           'f1_Str. Def', 'f1_TD Avg.', 'f1_TD Acc.', 'f1_TD Def.', 'f1_Sub. Avg.','f1_Reach','f2_Height_cm','f2_Reach', 'f2_SLpM',
           'f2_Str. Acc.', 'f2_SApM', 'f2_Str. Def', 'f2_TD Avg.', 'f2_TD Acc.',
           'f2_TD Def.', 'f2_Sub. Avg.']]

    male_fights = data[data['Weight class']>=4]

    #determining the percentage of knockouts in mens and womens weight classes
    print(data['Method_KO/TKO'].value_counts(normalize=True))
    print(male_fights.groupby('Weight class')['Method_KO/TKO'].mean())
    print(female_fights.groupby('Weight class')['Method_KO/TKO'].mean())

    #determining the percentage of submissions in mens and womens weight classes
    print(data['Method_SUB'].value_counts(normalize=True))
    print(male_fights.groupby('Weight class')['Method_SUB'].mean())
    print(female_fights.groupby('Weight class')['Method_SUB'].mean())

    #plot results
    labels = ['115', '125', '135', '145', '155','170','185', '205','265']
    men_ko = [0,22.3, 27.4, 25.81, 26.37, 32.53, 36.93, 45,55 ]
    women_ko = [9.7, 17.8, 25, 35, 0,0,0,0,0]
    x = np.arange(len(labels))
    width = 0.35
    fig2, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, men_ko, width, label='Men')
    rects2 = ax.bar(x + width / 2, women_ko, width, label='Women')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage Of Knockouts By Weight Class')
    ax.set_title('Weight Class')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    fig2.tight_layout()
    plt.show()

    #submission
    men_sub = [0, 18, 23, 17, 22, 17, 20, 15, 13]
    women_sub = [21, 20, 14, 24, 0, 0, 0, 0, 0]
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig2, ax = plt.subplots()
    rects3 = ax.bar(x - width / 2, men_sub, width, label='Men')
    rects4 = ax.bar(x + width / 2, women_sub, width, label='Women')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage Of Submissions By Weight Class')
    ax.set_title('Weight Class')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)
    fig2.tight_layout()
    plt.show()

    #womens accuracy
    womens_X,womens_y = data_split_with_processing(female_fights)
    mens_X, mens_y = data_split_with_processing(male_fights)
    model = RandomForestClassifier(criterion='entropy', max_depth=10, max_features=None,
                                                          min_samples_leaf=10, min_samples_split=10,
                                                          n_estimators=100)
    womens_scores = cross_val_score(model,womens_X,womens_y, cv=10, scoring='accuracy')
    print(womens_scores)
    print("Mean Accuracy womens scores: "+ str(np.mean(womens_scores)) + " St.Dev: "+ str(np.std(womens_scores)))

    model.fit(womens_X,womens_y)

    ft_imprtnces = pd.Series(model.feature_importances_, index=data.columns[:-1])
    ft_imprtnces = ft_imprtnces.nlargest(10).plot(kind='bar')
    plt.title('Top 10 Important Features - Womens')
    plt.tight_layout()
    plt.show()

    #mens accuracy
    model2 = RandomForestClassifier(criterion='entropy', max_depth=10, max_features=None,
                                   min_samples_leaf=10, min_samples_split=10,
                                   n_estimators=100)
    mens_scores = cross_val_score(model2, mens_X, mens_y, cv=10, scoring='accuracy')
    print(mens_scores)
    print("Mean Accuracy Mens scores: " + str(np.mean(mens_scores)) + " St.Dev: "+ str(np.std(mens_scores)))
    model2.fit(mens_X,mens_y)

    ft_imprtnces2 = pd.Series(model2.feature_importances_, index=data.columns[:-1])
    ft_imprtnces2 = ft_imprtnces2.nlargest(10).plot(kind='bar')
    plt.title('Top 10 Important Features- Mens')
    plt.tight_layout()
    plt.show()

    #statistical analysis of accuracies
    t_stat, p_val = ttest_ind(mens_scores,womens_scores)
    print("Test statistic: " + str(t_stat) + " P value: " + str(p_val))
    if p_val<= 0.05:
        print('We reject null hypothesis')
    else:
        print('We do not reject the null hypothesis')











if __name__ == '__main__':
    main()



