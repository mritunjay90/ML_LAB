# Lab 5 Report

## Title
Breast Cancer Classification with Logistic Regression

## Objective
The goal of this lab is to build a machine learning classification model using the breast cancer dataset from scikit-learn. The model is trained to predict whether a tumor is malignant or benign based on the provided features.

## Dataset
- Source: scikit-learn breast cancer dataset
- Features: numerical measurements of tumor characteristics
- Target: diagnosis labels (malignant/benign)

## Tools and Libraries
- Python
- scikit-learn
- pandas
- numpy
- matplotlib

## Methodology
1. Load the dataset.
2. Split the data into training and testing sets.
3. Standardize the feature values.
4. Train a Logistic Regression model.
5. Evaluate the model using accuracy, confusion matrix, and classification report.

## Results
The trained logistic regression model was evaluated on the test set with the following metrics:
- Accuracy: 0.9737
- Precision: 0.9722
- Recall: 0.9859
- F1-score: 0.9790
- Confusion Matrix:
  - [[41, 2], [1, 70]]

The model coefficients showed that:
- Largest positive coefficient: compactness error
- Largest negative coefficient: worst texture

## Conclusion
This lab demonstrates the application of logistic regression for binary classification using the Breast Cancer dataset. It highlights the importance of data preprocessing, train/test splitting, model training, and performance evaluation in machine learning.

## Notes
- The notebook requires scikit-learn and other standard Python machine learning libraries.
- The notebook can be opened and executed in Jupyter or VS Code.
