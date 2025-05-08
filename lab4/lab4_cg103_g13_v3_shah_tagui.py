import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# Import models (logistic regression, KNN)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

import random

def set_seed(seed):
    seed = 0
    random.seed(seed)

if __name__ == "__main__":
    # Set seed for reproducibility
    seed = 0
    set_seed(seed)

    # Load the iris dataset
    iris_dataset = load_iris()
    X = pd.DataFrame(data=iris_dataset.data, columns=iris_dataset.feature_names) # Columns: sepal length (cm), sepal width (cm), petal length (cm), petal width (cm)
    y = pd.Series(data=iris_dataset.target, name='target') # Target: 0 = setosa, 1 = versicolor, 2 = virginica

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Preprocess - Standardise features
    scaler = StandardScaler()
    scaler_train = scaler.fit(X_train) # Fit scaler on training data only to avoid data leakage
    X_train = scaler_train.transform(X_train) # Transform training data
    X_test = scaler_train.transform(X_test) # Transform test data using the same scaler

    # Define the models
    num_neighbours = 5 # Number of neighbors for KNN
    num_iter = 2 # Number of iterations for Logistic Regression
    model1 = LogisticRegression(random_state=seed, max_iter = num_iter) # Logistic Regression model
    model2 = KNeighborsClassifier(n_neighbors=num_neighbours) # KNN Classifier

    # Evaluate model using cross-validation
    scores1 = cross_val_score(model1, X_train, y_train, cv=4, scoring="accuracy")
    scores2 = cross_val_score(model2, X_train, y_train, cv=4, scoring="accuracy")

    print(f"Cross-validation scores for Logistic Regression: {scores1}")
    print(f"Cross-validation scores for KNN: {scores2}")

    # Fit the best model on the entire training set and get the predictions
    final_model1 = model1.fit(X_train, y_train)
    final_model2 = model2.fit(X_train, y_train)

    predictions1 = final_model1.predict(X_test)
    predictions2 = final_model2.predict(X_test)

    # Evaluate the final predictions with the metric of your choice (accuracy)
    accuracy1 = np.mean(predictions1 == y_test) * 100
    accuracy2 = np.mean(predictions2 == y_test) * 100
    print(f"Accuracy of Logistic Regression: {accuracy1:.2f}%")
    print(f"Accuracy of KNN: {accuracy2:.2f}%")
