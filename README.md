# Credit Card Fraud Detection
This repository contains code and results for a credit card fraud detection project that utilizes both traditional machine learning (ML) algorithms and deep learning neural networks. The goal of this project is to determine whether a credit card transaction is fraudulent or not based on various features and characteristics of the transaction data.

## Table of Contents
- [Overview](#overview)
- [Machine Learning Results](#machine-learning-results)
- [Deep Learning Results](#deep-learning-results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [License](#license)


## Overview
Credit card fraud detection is a critical task in the financial industry to protect customers and institutions from unauthorized transactions. This project explores two different approaches to tackle this problem: traditional machine learning algorithms and deep learning neural networks.

## Machine Learning Results
The machine learning model results are as follows:

Training Data Accuracy: Approximately 94.16%

Test Data Accuracy: Approximately 93.40%

Confusion Matrix:

```
Predicted    0    1    All
Actual
0           96    3    99
1           10   88    98
All        106   91   197
```

***Note: The confusion matrix is "flipped" because the focus is on predicting "fraudulent" transactions (class 1). This means that Type I errors (false positives) are of particular concern.***

## Deep Learning Results
The deep learning neural network model results are as follows:

Confusion Matrix:
```
Actual         Fraud      Normal
Predicted
Fraud          56845          16
Normal            18          83
```

Area Under the Precision-Recall Curve (AUPRC): 0.6893

Area Under the Receiver Operating Characteristic Curve (AUROC): 0.9108

Accuracy Score: 99.94%

Precision, Recall, and F1-Score:

```
           precision    recall  f1-score   support

        0       1.00      1.00      1.00     56861
        1       0.84      0.82      0.83       101

 accuracy                           1.00     56962
macro avg       0.92      0.91      0.91     56962
weighted avg       1.00      1.00      1.00     56962
```

## Repository Structure
```
| Credit Card Fraud Detection ML.ipynb: Jupyter Notebook containing the code and results for the machine learning approach.
| Credit Card Fraud Detection DL.ipynb: Jupyter Notebook containing the code and results for the deep learning approach.
| data/: Directory containing the dataset used for training and testing.
| README.md: This documentation file.
| LICENSE: The license for this repo
| Pipfile: Pipfile for environment
| requirements.txt: The required dependencies required to run this repo.
```

## Getting Started
To run or reproduce the results presented in the notebooks, follow these steps:

1. Clone this repository to your local machine.
2. Start a virtual environment
3. Install the required dependencies: `pip3 install -r requirements.txt`
4. Start Jupyter Lab: `jupyter lab`

Customize and fine-tune the models or experiment with different hyperparameters and feature engineering techniques to improve performance further.

## License
This project is licensed under the MIT License. See the LICENSE file for details.