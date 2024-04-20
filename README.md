# Heart-Disease-Prediction-Using-Python

This repository contains Python scripts for predicting heart disease using various machine learning models: Decision Tree, Random Forest, and Deep Neural Network (DNN). Each script loads a dataset, preprocesses the data, trains a specific machine learning model, and evaluates its performance.


# Heart Disease Prediction using Machine Learning Models

This repository contains Python scripts for predicting heart disease using various machine learning models: Decision Tree, Random Forest, and Deep Neural Network (DNN). Each script loads a dataset, preprocesses the data, trains a specific machine learning model, and evaluates its performance.

## Decision Tree

The script `decision_tree.py` implements a Decision Tree classifier. It loads a dataset from a CSV file, splits it into features and target, then splits the data into training and testing sets. The Decision Tree classifier is trained on the training data, and predictions are made on the test set. The accuracy of the model is then evaluated.

## Random Forest

The script `random_forest.py` implements a Random Forest classifier. Similar to the Decision Tree script, it loads the dataset, splits it, and trains a Random Forest classifier. Predictions are made on the test set, and the accuracy of the model is evaluated.

## Deep Neural Network (DNN)

The script `dnn.py` builds a DNN model using Keras. It loads the dataset, splits it into training and testing sets, and constructs a DNN model with multiple dense layers. The model is trained on the training data and evaluated on the test set. Additionally, the overall confidence average of the predictions is calculated.

## Dependencies

All scripts require the following libraries:
- pandas
- numpy
- scikit-learn
- matplotlib
- keras
