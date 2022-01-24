## Machine Learning and Mixed Martial Arts

This project aims to produce models that can predict the winner of Mixed Martial Arts fights using Machine Learning and
Data Science techniques. An autoencoder is also created to perform feature extraction and dimensionality reduction to
increase the accuracy of the model. Further work will compare differences in the machine learning models ability to
predict the difference in accuracy on men's and women's data sets as well as a comparison of feature importance and
winning methods.

## Configuration information - How to run:

This project was developed and ran in Python 3.9.1

## Dependencies - 3rd Party Libraries

All software libraries:

- pandas version 1.2.4 - Python library for data analysis and manipulation tool
- NumPy version 1.19.5 - Python library for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays,
- matplotlib version 3.4.2 - Python library for for creating static, animated, and interactive visualizations in Python
- BeautifulSoup4 version 4.9.3 -Python library that makes it easy to scrape information from web pages, 
- TensorFlow version 2.6.0 - Python library for machine learning and artificial intelligence
- scikit-learn version 0.24.2 - Python library for machine learning and artificial intelligence
- imblearn version 0.0 - Python library with tools for classification with imbalanced classes
- MLXtend version 0.18.0  - Python library with tools for common data science practices
- SciPy version 1.7.0- Python library used for scientific computing and technical computing.
- GraphViz version 0.16 - Python library that is used to create graph objects which can be completed using different nodes and edges 
- xgboost version 1.4.2 - an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install relevant library.

```bash
pip install <'libraryname'>
```

## File list

* model_from_scratch 
    * K_Nearest_Neighbour.py - Model developed from scratch using numpy, includes hyperparameter tuning
    * Logistic_regression.py - Model developed from scratch using numpy, includes hyperparameter tuning
    * SVM_SMO.py - Model developed from scratch using numpy, includes hyperparameter tuning
    * Scratch_Model_Performance.py - analysis of from scratch models versus those developed by sklearn

* sklearn_ensemble_and_complex_methods
    * AdaBoostClassifier.py - Model developed from sklearn library includes hyperparameter tuning
    * NeuralNetwork.py - Model developed from sklearn library includes hyperparameter tuning
    * Random_Forest_sklearn.py - Model developed from sklearn library includes hyperparameter tuning
    * XGBoost.py - Model developed from sklearn library includes hyperparameter tuning

* sklearn_traditional_methods 
    * Decision_Tree_sklearn.py - Model developed from sklearn library includes hyperparameter tuning
    * KNN_sklearn.py - Model developed from sklearn library includes hyperparameter tuning
    * Logistic_Regression_sklearn.py - Model developed from sklearn library includes hyperparameter tuning
    * SVM_sklearn.py - Model developed from sklearn library includes hyperparameter tuning

* Autoencoder.py - autoencoder developed for feature extraction
* Comparison_mens_womens.py - analysis of womens and mens data sets and the ability to predict the winner of both
* encoded_data_classification.py - classification on the encoded data set
* FighterStatProcessing.py - Cleans and processes the data from the webscrapers
* encoder.h5 - exported encoder from Autoencoder.py, used in encoded_data_classification.py
* Statistical_Analysis_Models.py - analysis of all sklearn models
* match_scraper.py - web scraper that scrapes fight data from ufcstats.com
* ScrapingFighterStats.py-  - web scraper that scrapes fighter data from ufcstats.com

* CSV Files
   * final_data.csv- final pre-processed data, imported into model and analysis files using pandas
   * newfightinfo.csv - web scraped fight data
   * practiceallfighterinfo.csv - web scraped fighter data

## Licenses

- pandas -BSD 3-Clause License

- NumPy - numpy/numpy is licensed under the

BSD 3-Clause "New" or "Revised" License
A permissive license similar to the BSD 2-Clause License, but with a 3rd clause that prohibits others from using the name of the project or its contributors to promote derived products without written consent.

- matplotlib - https://github.com/matplotlib/matplotlib/blob/master/LICENSE/LICENSE
- BeautifulSoup4 - The MIT License (MIT) https://github.com/akalongman/python-beautifulsoup/blob/master/LICENSE
- TensorFlow - tensorflow/tensorflow is licensed under the

Apache License 2.0
A permissive license whose main conditions require preservation of copyright and license notices. Contributors provide an express grant of patent rights. Licensed works, modifications, and larger works may be distributed under different terms and without source code.

- scikit-learn - scikit-learn/scikit-learn is licensed under the

BSD 3-Clause "New" or "Revised" License
A permissive license similar to the BSD 2-Clause License, but with a 3rd clause that prohibits others from using the name of the project or its contributors to promote derived products without written consent.

- imblearn - scikit-learn-contrib/imbalanced-learn is licensed under the

MIT License
A short and simple permissive license with conditions only requiring preservation of copyright and license notices. Licensed works, modifications, and larger works may be distributed under different terms and without source code.

- MLXtend - New BSD License

Copyright (c) 2014-2020, Sebastian Raschka. All rights reserved.

- SciPy- BSD New License

- GraphViz dot - https://github.com/ppareit/graphviz-dot-mode/blob/master/LICENSE.md 


- xgboost - https://github.com/dmlc/xgboost/blob/master/LICENSE

## References - libraries used in code for all files

- Beautiful Soup 4 - Richardson, Leonard (2007). “Beautiful soup documentation”. In: April.


- pandas -McKinney, Wes (2010). “Data Structures for Statistical Computing in Python”. In: Proceedings of the 9th
Python in Science Conference. Ed. by Stéfan van der Walt and Jarrod Millman, pp. 56–61. doi:
10.25080/Majora-92bf1922-00a

- NumPy - Harris, Charles R. et al. (Sept. 2020). “Array programming with NumPy”. In: Nature 585.7825, pp. 357–362.
doi: 10.1038/s41586-020-2649-2. url: https://doi.org/10.1038/s41586-020-2649-2.


- Matplotlib - Hunter, J. D. (2007). “Matplotlib: A 2D graphics environment”. In: Computing in Science & Engineering
9.3, pp. 90–95. doi: 10.1109/MCSE.2007.55.

- TensorFlow - Martın Abadi et al. (2015). TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems. Software
available from tensorflow.org. url: https://www.tensorflow.org/

- Scikit-Learn - Pedregosa, F. et al. (2011). “Scikit-learn: Machine Learning in Python”. In: Journal of Machine Learning
Research 12, pp. 2825–2830

- imblearn - Lemaître, Guillaume, Fernando Nogueira, and Christos K. Aridas (2017). “Imbalanced-learn: A Python
Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning”. In: Journal of Machine
Learning Research 18.17, pp. 1–5. url: http://jmlr.org/papers/v18/16-365.html

- mlxtend - Raschka, Sebastian (Apr. 2018). “MLxtend: Providing machine learning and data science utilities and extensions to Python’s scientific computing stack”. In: The Journal of Open Source Software 3.24. doi:
10.21105/joss.00638. url: http://joss.theoj.org/papers/10.21105/joss.00638.

- scipy - Virtanen, Pauli et al. (2020). “SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python”. In:
Nature Methods 17, pp. 261–272. doi: 10.1038/s41592-019-0686-2.


## References for code and research used - also in Files

- Logistic_regression.py -

  - Title: Logistic Regression in Python
  - Author: Python Engineer
  - Availability: https://www.youtube.com/watch?v=JDU3AzH3WKg&t=2s
  - Date: 15 Sept 2019

  - Title: Logistic Regression in Python from Scratch
  - Author: Coding Lane
  - Availability: https://www.youtube.com/watch?v=nzNp05AyBM8
  - Date: 4 Feb 2021

  - Title: Logistic Regression from Scratch - Machine Learning Python
  - Author: Aladdin Persson
  - Availability: https://www.youtube.com/watch?v=x1ez9vi611I
  - Date: 29 May 2020

- K_Nearest_Neighbour.py -

  - Title: KNN (K Nearest Neighbours) in Python
  - Author: Python Engineer
  - Availability: https://www.youtube.com/watch?v=ngLyX54e1LU&t=1130s
  - Date: 3 Sept 2019

  - Title: K Nearest Neighbors Application - Practical Machine Learning Tutorial with Python p.14
  - Author: sentdex
  - Availability: https://www.youtube.com/watch?v=1i0zu9jHN6U&ab_channel=sentdex
  - Date: 1 May 2016

  - Title: Creating Our K Nearest Neighbors Algorithm - Practical Machine Learning with Python p.16
  - Author: sentdex
  - Availability: https://www.youtube.com/watch?v=1i0zu9jHN6U&ab_channel=sentdex
  - Date: 5 May 2016

  - Title: K Nearest Neighbour from Scratch - Machine Learning Python
  - Author: Aladdin Persson
  - Availability: https://www.youtube.com/watch?v=QzAaRuDskyc&t=559s
  - Date: 24 Apr 2020

- SVM_SMO.py -

  - Title: simplest_smo_ever/simple_svm.ipynb
  - Author: Tymchyshyn and Khavliuk
  - Date: 25/02/21
  - Code Version:1
  - Availability: https://github.com/fbeilstein/simplest_smo_ever/blob/master/simple_svm.ipynb

- FighterStatProcessing.py - see code for section where below was used -

  - Title: preprocessing.ipynb
  - Author: Yuan Tian
  - Date: 15/09/2020
  - Code Version : 1.0
  - Availability: https://github.com/naity/DeepUFC2/blob/master/preprocessing.ipynb

- SVM_SMO.py
  - Title: simplest_smo_ever/simple_svm.ipynb
  - Author: Tymchyshyn and Khavliuk
  - Date: 25/02/21
  - Code Version:1
  - Availability: https://github.com/fbeilstein/simplest_smo_ever/blob/master/simple_svm.ipynb

- Autoencoder.py
  - Title: Autoencoder Feature Extraction for Classification
  - Author: Jason Brownlee
  - Date: 12/07/20
  - Code version: 1.0
  - Availability: https://machinelearningmastery.com/autoencoder-for-classification/



