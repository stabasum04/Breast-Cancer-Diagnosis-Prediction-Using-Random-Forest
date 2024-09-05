# Breast-Cancer-Diagnosis-Prediction-Using-Random-Forest

This repository contains a machine learning project aimed at predicting the diagnosis of breast cancer (benign or malignant) using the Random Forest algorithm. The dataset used is the "Breast Cancer Wisconsin (Diagnostic) Data Set," and the project demonstrates the entire workflow .

#Project Overview
Breast cancer is one of the most common cancers in women worldwide. Early and accurate diagnosis is crucial for successful treatment and patient survival. This project utilizes a Random Forest Regressor model to predict whether a tumor is benign (non-cancerous) or malignant (cancerous) based on several input features extracted from digitized images of fine needle aspirates (FNAs) of breast mass.


Number of instances: 569

Number of attributes: 32 (ID, diagnosis, 30 real-valued input features)

Diagnosis (M = malignant, B = benign)

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter)
b) texture (standard deviation of gray-scale values)
c) perimeter
d) area
e) smoothness (local variation in radius lengths)
f) compactness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)
i) symmetry
j) fractal dimension ("coastline approximation" - 1)

Missing attribute values: none

Class distribution: 357 benign, 212 malignant




#Features Used
The model is trained using the following features from the dataset:

radius_mean: Mean of distances from the center to points on the perimeter
perimeter_mean: Mean size of the core tumor
texture_mean: Standard deviation of gray-scale values
area_mean: Mean area of the tumor
smoothness_mean: Mean of local variation in radius lengths
concavity_mean: Mean of severity of concave portions of the contour
symmetry_mean: Mean of symmetry of the tumor
perimeter_se: Standard error of the perimeter
concave points_se: Standard error of concave points
fractal_dimension_se: Standard error of fractal dimension ("coastline approximation" - 1)
area_worst: Largest mean area of the tumor
fractal_dimension_worst: Worst or largest value for fractal dimension


#Project Workflow
Data Preprocessing:

Reading the data from CSV file.
Cleaning and preparing the dataset for training.
Encoding the target variable (Diagnosis) into numerical values for model compatibility.
Model Training and Evaluation:

The dataset is split into training and testing sets using train_test_split.
A RandomForestRegressor model is created and trained on the training set.
Model performance is evaluated using R-squared scores.
Decision Tree Visualization:

One of the decision trees from the trained Random Forest model is visualized to understand feature splits and importance.
Interactive Prediction:

An interactive Python function allows users to input values for all features and receive a prediction (0 for Benign, 1 for Malignant).
Dependencies
Python 3.x
pandas
numpy
scikit-learn(randomforestregressor)
matplotlib

