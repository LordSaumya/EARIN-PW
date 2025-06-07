import polars as pl
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from time import perf_counter_ns
from typing import Tuple, Dict, Any
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random

# Constants
TARGET_COLUMN = "output"  # Target column name

CONTINUOUS_COLUMNS = ["age", "trtbps", "chol", "thalachh", "oldpeak"]

ORDINAL_COLUMNS = ["cp", "restecg", "slp", "caa", "thall"]

NOMINAL_COLUMNS = [
    "sex",
    "fbs",
    "exng",
    # Output is nominal but excluded from features
]

DISCRETE_COLUMNS = ORDINAL_COLUMNS + NOMINAL_COLUMNS  # All columns with discrete values

INTERNAL_PLOTS = True  # Show internal plots during model training
COMPARISON_PLOTS = True  # Show comparison plots for models


def gower_distance(
    X1: np.ndarray, X2: np.ndarray, feature_info: Dict[str, Any]
) -> np.ndarray:
    n1, n2 = X1.shape[0], X2.shape[0]
    distances = np.zeros((n1, n2))

    continuous_indices = feature_info["continuous_indices"]
    ordinal_indices = feature_info["ordinal_indices"]
    nominal_indices = feature_info["nominal_indices"]
    ranges = feature_info["ranges"]

    for i in range(n1):
        for j in range(n2):
            total_distance = 0.0
            valid_comparisons = 0

            # Continuous/ordinal features: range-normalised absolute difference
            for idx in continuous_indices + ordinal_indices:
                if ranges[idx] > 0:  # Avoid division by zero
                    diff = abs(X1[i, idx] - X2[j, idx]) / ranges[idx]
                    total_distance += diff
                    valid_comparisons += 1

            # Nominal features: simple matching (0 if same, 1 if different)
            for idx in nominal_indices:
                diff = 0.0 if X1[i, idx] == X2[j, idx] else 1.0
                total_distance += diff
                valid_comparisons += 1

            # Average distance across all valid comparisons
            distances[i, j] = (
                total_distance / valid_comparisons if valid_comparisons > 0 else 0.0
            )

    return distances


def preprocess_data(df: pl.LazyFrame) -> pl.DataFrame:
    # Classes are relatively balanced, so no need for up/down sampling or SMOTE
    # There are no missing values in the dataset, so no need for imputation
    # All nominal columns are already encoded as integers, so no need for label encoding
    # PCA is not needed as the dataset is small and interpretable, and correlations between features is not significant

    initial_shape = (
        df.select(pl.len()).collect().item(),
        len(df.collect_schema().names()),
    )

    # Remove outliers based on IQR
    def remove_outliers_iqr_lazy(df: pl.LazyFrame, column: str) -> pl.LazyFrame:
        # Calculate IQR and bounds
        q1 = df.select(pl.col(column).quantile(0.25)).collect().item()
        q3 = df.select(pl.col(column).quantile(0.75)).collect().item()
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        filtered_df = df.filter(
            (pl.col(column) >= lower_bound) & (pl.col(column) <= upper_bound)
        )
        return filtered_df

    # Remove outliers for numerical columns
    columns_to_check = CONTINUOUS_COLUMNS

    for column in columns_to_check:
        df = remove_outliers_iqr_lazy(df, column)

    # Find highly correlated features
    def get_features_to_drop_by_correlation(
        df_lazy: pl.LazyFrame, target_column_name: str, threshold: float
    ) -> tuple[list[str], dict]:
        df_collected = df_lazy.collect()

        # Calculate correlation with target column for each feature
        correlated_features = []
        correlations = {}

        for col in df_collected.columns:
            if col != target_column_name:
                # Calculate correlation between column and target
                corr_result = df_collected.select(
                    [
                        pl.corr(pl.col(col), pl.col(target_column_name)).alias(
                            "correlation"
                        )
                    ]
                ).item()

                correlations[col] = corr_result

                if abs(corr_result) > threshold:
                    correlated_features.append(col)

        return correlated_features, correlations

    features_to_drop, _ = get_features_to_drop_by_correlation(
        df_lazy=df, target_column_name=TARGET_COLUMN, threshold=0.8
    )

    # Drop highly correlated features
    if features_to_drop:
        df = df.drop(features_to_drop)

    final_shape = (
        df.select(pl.len()).collect().item(),
        len(df.collect_schema().names()),
    )

    print(
        f"Initial shape: {initial_shape}, Final shape after preprocessing: {final_shape}"
    )
    return df.collect()


def split_data(
    df: pl.DataFrame, test_size: float, validation_size: float
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    # Shuffle the DataFrame
    df = df.sample(fraction=1.0, shuffle=True, seed=42)

    # Calculate sizes
    total_size = df.shape[0]
    test_size = int(total_size * test_size)
    validation_size = int(total_size * validation_size)

    # Split into test set
    test_set = df.head(test_size)

    # Remaining data for training and validation
    remaining_data = df.tail(total_size - test_size)

    # Split remaining data into training and validation sets
    validation_set = remaining_data.head(validation_size)
    train_set = remaining_data.tail(remaining_data.shape[0] - validation_size)

    return train_set, validation_set, test_set


# Function for KNN-specific preprocessing with Gower distance
def preprocess_for_knn(
    df: pl.DataFrame,
    existing_feature_info: Dict[str, Any] = None,
    target_column_name: str = TARGET_COLUMN,
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    # No scaling is required for Gower distance, but we need to prepare feature information

    df_copy = df.clone()

    # Get all feature columns (exclude target)
    all_features = [col for col in df_copy.columns if col != target_column_name]

    if existing_feature_info is None:
        # Compute feature information on training data for Gower distance
        feature_info = {
            "continuous_indices": [],
            "ordinal_indices": [],
            "nominal_indices": [],
            "ranges": {},
            "feature_names": all_features,
        }

        # Map column names to indices
        for i, col in enumerate(all_features):
            if col in CONTINUOUS_COLUMNS:
                feature_info["continuous_indices"].append(i)
            elif col in ORDINAL_COLUMNS:
                feature_info["ordinal_indices"].append(i)
            elif col in NOMINAL_COLUMNS:
                feature_info["nominal_indices"].append(i)

        # Compute ranges for continuous and ordinal features for normalization
        data_array = df_copy.select(all_features).to_numpy()
        for i, col in enumerate(all_features):
            if col in CONTINUOUS_COLUMNS or col in ORDINAL_COLUMNS:
                col_data = data_array[:, i]
                feature_info["ranges"][i] = np.max(col_data) - np.min(col_data)
            else:
                # For nominal features, range is not needed (matching distance)
                feature_info["ranges"][i] = 1.0

        print("Feature mapping for Gower distance:")
        print(
            f"  Continuous features: {[all_features[i] for i in feature_info['continuous_indices']]}"
        )
        print(
            f"  Ordinal features: {[all_features[i] for i in feature_info['ordinal_indices']]}"
        )
        print(
            f"  Nominal features: {[all_features[i] for i in feature_info['nominal_indices']]}"
        )

    else:
        # Use existing feature info for validation/test data
        feature_info = existing_feature_info

    return df_copy, feature_info


def knn_model(train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: pl.DataFrame):
    # Preprocess data for KNN with Gower distance
    train_df_knn, feature_info = preprocess_for_knn(train_df.clone())
    val_df_knn, _ = preprocess_for_knn(
        val_df.clone(), existing_feature_info=feature_info
    )
    test_df_knn, _ = preprocess_for_knn(
        test_df.clone(), existing_feature_info=feature_info
    )

    # Select all features except the target column
    features_for_model = [col for col in train_df_knn.columns if col != TARGET_COLUMN]

    X_train = train_df_knn.select(features_for_model).to_numpy()
    y_train = train_df_knn.select(TARGET_COLUMN).to_numpy().flatten()
    X_val = val_df_knn.select(features_for_model).to_numpy()
    y_val = val_df_knn.select(TARGET_COLUMN).to_numpy().flatten()
    X_test = test_df_knn.select(features_for_model).to_numpy()
    y_test = test_df_knn.select(TARGET_COLUMN).to_numpy().flatten()

    # Custom KNN prediction function using Gower distance
    def predict_knn_gower(X_train, y_train, X_test, k, feature_info):
        n_test = X_test.shape[0]
        predictions = np.zeros(n_test)

        for i in range(n_test):
            # Compute distances from test point to all training points
            X_test_point = X_test[i : i + 1]  # Keep as 2D array
            distances = gower_distance(X_test_point, X_train, feature_info)[0]

            # Find k nearest neighbors
            nearest_indices = np.argsort(distances)[:k]
            nearest_labels = y_train[nearest_indices]

            # Majority vote
            predictions[i] = np.bincount(nearest_labels.astype(int)).argmax()

        return predictions.astype(int)

    # Custom KNN probability prediction function using Gower distance used for ROC curve
    def predict_prob_knn_gower(X_train, y_train, X_test, k, feature_info):
        n_test = X_test.shape[0]
        n_classes = len(np.unique(y_train))
        probabilities = np.zeros((n_test, n_classes))

        for i in range(n_test):
            # Compute distances from test point to all training points
            X_test_point = X_test[i : i + 1]  # Keep as 2D array
            distances = gower_distance(X_test_point, X_train, feature_info)[0]

            # Find k nearest neighbors
            nearest_indices = np.argsort(distances)[:k]
            nearest_labels = y_train[nearest_indices].astype(int)

            # Calculate class probabilities
            for class_label in range(n_classes):
                probabilities[i, class_label] = (
                    np.sum(nearest_labels == class_label) / k
                )

        return probabilities

    # Hyperparameter tuning for n_neighbors using F1 score on validation set
    k_values = list(range(1, 21))  # Testing k from 1 to 20
    best_k = k_values[0]
    best_f1_val = 0.0
    validation_f1_scores = []  # Store F1 scores for plotting

    print("Starting hyperparameter tuning for KNN...")
    for k in k_values:
        y_val_pred_temp = predict_knn_gower(X_train, y_train, X_val, k, feature_info)
        f1_val_temp = f1_score(y_val, y_val_pred_temp)
        validation_f1_scores.append(f1_val_temp)
        print(f"  k={k}: Validation F1 Score = {f1_val_temp:.4f}")
        if f1_val_temp > best_f1_val:
            best_f1_val = f1_val_temp
            best_k = k

    print(f"Best k found: {best_k} with Validation F1 Score: {best_f1_val:.4f}")

    # Plot k_values vs validation_f1_scores
    if INTERNAL_PLOTS:
        fig = px.line(
            x=k_values,
            y=validation_f1_scores,
            labels={"x": "k (Number of Neighbors)", "y": "Validation F1 Score"},
            title="KNN Hyperparameter Tuning: k vs. Validation F1 Score",
        )
        fig.show()

    # Evaluate final model on validation and test sets using best k

    # Time the validation prediction
    start_val_time = perf_counter_ns()
    y_val_pred_final = predict_knn_gower(X_train, y_train, X_val, best_k, feature_info)
    end_val_time = perf_counter_ns()
    val_prediction_time = (end_val_time - start_val_time) / 1e9  # Convert to seconds

    f1_val_final = f1_score(y_val, y_val_pred_final)

    # Time the test prediction
    start_test_time = perf_counter_ns()
    y_test_pred_final = predict_knn_gower(
        X_train, y_train, X_test, best_k, feature_info
    )
    end_test_time = perf_counter_ns()
    test_prediction_time = (end_test_time - start_test_time) / 1e9  # Convert to seconds

    f1_test_final = f1_score(y_test, y_test_pred_final)

    # Summary
    print("\n\n ## Summary for tuned KNN model ##")
    print(f"Final KNN Model - Validation F1 Score (k={best_k}): {f1_val_final:.4f}")
    print(f"Final KNN Model - Test F1 Score (k={best_k}): {f1_test_final:.4f}")
    print(f"Validation prediction:   {val_prediction_time:.8f} seconds")
    print(f"Test prediction:         {test_prediction_time:.8f} seconds")

    # Calculate samples per second
    val_samples_per_sec = len(X_val) / val_prediction_time
    print(f"Samples per second (val): {val_samples_per_sec:.0f} samples/sec")

    test_samples_per_sec = len(X_test) / test_prediction_time
    print(f"Samples per second (test): {test_samples_per_sec:.0f} samples/sec")

    # Calculate total samples per second
    total_samples = len(X_val) + len(X_test)
    total_inference_time = val_prediction_time + test_prediction_time
    total_samples_per_sec = total_samples / total_inference_time
    print(f"Samples per second (total): {total_samples_per_sec:.0f} samples/sec")

    print("Note: KNN does not have a training phase")

    # Confusion Matrix for Test Set
    if INTERNAL_PLOTS:
        cm = confusion_matrix(y_test, y_test_pred_final)
        cm_fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["No Risk (0)", "Risk (1)"],
            y=["No Risk (0)", "Risk (1)"],
            text_auto=True,
            title=f"Gower KNN Confusion Matrix for Test Set (k={best_k})",
            color_continuous_scale=px.colors.sequential.Blues,
        )
        cm_fig.update_layout(xaxis_title="Predicted Label", yaxis_title="True Label")
        cm_fig.show()

    # ROC Curve and AUC for Test Set
    y_test_prob = predict_prob_knn_gower(
        X_train, y_train, X_test, best_k, feature_info
    )[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
    roc_auc = auc(fpr, tpr)

    if INTERNAL_PLOTS:
        roc_fig = go.Figure()
        # Plot ROC curve
        roc_fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"Gower KNN ROC curve (AUC = {roc_auc:.2f})",
            )
        )
        # Plot random chance line
        roc_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Chance (AUC = 0.5)",
                line=dict(dash="dash"),
            )
        )

        roc_fig.update_layout(
            title=f"Gower KNN Receiver Operating Characteristic (ROC) Curve for Test Set (k={best_k})",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=0.7, y=0.1),
        )
        roc_fig.show()

    # Return ROC data for combined plotting
    return fpr, tpr, roc_auc, f"KNN (k={best_k})"


def mixed_naive_bayes_model(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    target_column_name: str = TARGET_COLUMN,
):
    # Define feature categories

    # Continuous features for Gaussian NB
    continuous_features = [col for col in CONTINUOUS_COLUMNS if col in train_df.columns]

    # All discrete features (ordinal + nominal) for Multinomial NB
    discrete_features = [col for col in DISCRETE_COLUMNS if col in train_df.columns]

    # Get all feature columns (exclude target)
    all_features = [col for col in train_df.columns if col != target_column_name]

    # Prepare data
    X_train = train_df.select(all_features).to_numpy()
    y_train = train_df.select(target_column_name).to_numpy().flatten()
    X_val = val_df.select(all_features).to_numpy()
    y_val = val_df.select(target_column_name).to_numpy().flatten()
    X_test = test_df.select(all_features).to_numpy()
    y_test = test_df.select(target_column_name).to_numpy().flatten()

    # Get column indices for different feature types
    feature_names = all_features
    continuous_indices = [
        feature_names.index(col) for col in continuous_features if col in feature_names
    ]
    discrete_indices = [
        feature_names.index(col) for col in discrete_features if col in feature_names
    ]

    print("Feature distribution for Mixed Naive Bayes:")
    print(f"  Continuous features (Gaussian NB): {continuous_features}")
    print(f"  Discrete features (Multinomial NB): {discrete_features}")

    # Create custom mixed Naive Bayes classifier
    class MixedNaiveBayes:
        def __init__(self, var_smoothing_gaussian=1e-9, alpha_multinomial=1.0):
            self.var_smoothing_gaussian = var_smoothing_gaussian
            self.alpha_multinomial = alpha_multinomial
            self.gaussian_nb = None
            self.multinomial_nb = None
            self.continuous_indices = continuous_indices
            self.discrete_indices = discrete_indices

        def fit(self, X, y):
            if self.continuous_indices:
                self.gaussian_nb = GaussianNB(var_smoothing=self.var_smoothing_gaussian)
                self.gaussian_nb.fit(X[:, self.continuous_indices], y)

            if self.discrete_indices:
                self.multinomial_nb = MultinomialNB(alpha=self.alpha_multinomial)
                self.multinomial_nb.fit(X[:, self.discrete_indices], y)

            return self

        def predict_prob(self, X):
            n_samples = X.shape[0]
            n_classes = len(np.unique(y_train))

            # Initialise probabilities
            log_probs = np.zeros((n_samples, n_classes))

            # Get probabilities from Gaussian NB for continuous features
            if self.gaussian_nb is not None:
                gaussian_log_probs = np.log(
                    self.gaussian_nb.predict_proba(X[:, self.continuous_indices])
                    + 1e-10
                )
                log_probs += gaussian_log_probs

            # Get probabilities from Multinomial NB for discrete features
            if self.multinomial_nb is not None:
                multinomial_log_probs = np.log(
                    self.multinomial_nb.predict_proba(X[:, self.discrete_indices])
                    + 1e-10
                )
                log_probs += multinomial_log_probs

            # Convert back to probabilities and normalise
            probs = np.exp(log_probs)
            probs = probs / np.sum(probs, axis=1, keepdims=True)

            return probs

        def predict(self, X):
            probs = self.predict_prob(X)
            return np.argmax(probs, axis=1)

    # Hyperparameter tuning using grid search (alpha and var_smoothing)
    gaussian_var_smoothing_values = np.logspace(-12, -6, num=20)
    multinomial_alpha_values = np.logspace(-3, 2, num=20)

    best_params = {
        "var_smoothing": gaussian_var_smoothing_values[0],
        "alpha": multinomial_alpha_values[0],
    }
    best_f1_val = 0.0

    print("Starting hyperparameter tuning for Mixed Naive Bayes...")
    print(
        f"Testing {len(gaussian_var_smoothing_values)} var_smoothing values and {len(multinomial_alpha_values)} alpha values..."
    )

    # Grid search over hyperparameters
    total_combinations = len(gaussian_var_smoothing_values) * len(
        multinomial_alpha_values
    )
    current_combination = 0

    for vs_val in gaussian_var_smoothing_values:
        for alpha_val in multinomial_alpha_values:
            current_combination += 1

            # Create and train mixed model
            mixed_nb = MixedNaiveBayes(
                var_smoothing_gaussian=vs_val, alpha_multinomial=alpha_val
            )
            mixed_nb.fit(X_train, y_train)

            # Evaluate on validation set
            y_val_pred_temp = mixed_nb.predict(X_val)
            f1_val_temp = f1_score(y_val, y_val_pred_temp)

            if current_combination % 50 == 0 or f1_val_temp > best_f1_val:
                print(
                    f"  Progress: {current_combination}/{total_combinations} - "
                    f"var_smoothing={vs_val:.2e}, alpha={alpha_val:.2e}, "
                    f"F1={f1_val_temp:.4f}"
                )

            if f1_val_temp > best_f1_val:
                best_f1_val = f1_val_temp
                best_params["var_smoothing"] = vs_val
                best_params["alpha"] = alpha_val

    print("Best parameters found:")
    print(f"  var_smoothing (Gaussian): {best_params['var_smoothing']:.2e}")
    print(f"  alpha (Multinomial): {best_params['alpha']:.2e}")
    print(f"  Validation F1 Score: {best_f1_val:.4f}")

    # Train final classifier with best parameters
    final_mixed_nb = MixedNaiveBayes(
        var_smoothing_gaussian=best_params["var_smoothing"],
        alpha_multinomial=best_params["alpha"],
    )

    # Time the training phase
    start_train_time = perf_counter_ns()
    final_mixed_nb.fit(X_train, y_train)
    end_train_time = perf_counter_ns()
    training_time = (end_train_time - start_train_time) / 1e9  # Convert to seconds

    # Time the validation prediction
    start_val_time = perf_counter_ns()
    y_val_pred_final = final_mixed_nb.predict(X_val)
    end_val_time = perf_counter_ns()
    val_prediction_time = (end_val_time - start_val_time) / 1e9  # Convert to seconds

    f1_val_final = f1_score(y_val, y_val_pred_final)

    # Time the test prediction
    start_test_time = perf_counter_ns()
    y_test_pred_final = final_mixed_nb.predict(X_test)
    end_test_time = perf_counter_ns()
    test_prediction_time = (end_test_time - start_test_time) / 1e9  # Convert to seconds

    f1_test_final = f1_score(y_test, y_test_pred_final)

    # Summary
    print("\n\n ## Summary for Mixed Naive Bayes model ##")
    print(f"Final Mixed NB Model - Validation F1 Score: {f1_val_final:.4f}")
    print(f"Final Mixed NB Model - Test F1 Score: {f1_test_final:.4f}")
    print(f"Training time:           {training_time:.8f} seconds")
    print(f"Validation prediction:   {val_prediction_time:.8f} seconds")
    print(f"Test prediction:         {test_prediction_time:.8f} seconds")

    # Calculate samples per second
    val_samples_per_sec = len(X_val) / val_prediction_time
    print(f"Samples per second (val): {val_samples_per_sec:.0f} samples/sec")

    test_samples_per_sec = len(X_test) / test_prediction_time
    print(f"Samples per second (test): {test_samples_per_sec:.0f} samples/sec")

    # Calculate total samples per second
    total_samples = len(X_val) + len(X_test)
    total_inference_time = val_prediction_time + test_prediction_time
    total_samples_per_sec = total_samples / total_inference_time
    print(f"Samples per second (total): {total_samples_per_sec:.0f} samples/sec")

    # Confusion Matrix for Test Set
    if INTERNAL_PLOTS:
        cm_nb = confusion_matrix(y_test, y_test_pred_final)
        cm_fig_nb = px.imshow(
            cm_nb,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["No Risk (0)", "Risk (1)"],
            y=["No Risk (0)", "Risk (1)"],
            text_auto=True,
            title="Mixed NB Confusion Matrix for Test Set",
            color_continuous_scale=px.colors.sequential.Greens,
        )
        cm_fig_nb.update_layout(xaxis_title="Predicted Label", yaxis_title="True Label")
        cm_fig_nb.show()

    # ROC Curve and AUC for Test Set
    y_test_proba = final_mixed_nb.predict_prob(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)

    if INTERNAL_PLOTS:
        roc_fig = go.Figure()
        roc_fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"Mixed NB ROC curve (AUC = {roc_auc:.2f})",
            )
        )
        roc_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Chance (AUC = 0.5)",
                line=dict(dash="dash"),
            )
        )

        roc_fig.update_layout(
            title="Mixed NB Receiver Operating Characteristic (ROC) Curve for Test Set",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=0.7, y=0.1),
        )
        roc_fig.show()

    # Return ROC data for combined plotting
    return fpr, tpr, roc_auc, "Mixed Naive Bayes"


def neural_network_model(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    target_column_name: str = TARGET_COLUMN,
):
    # Get all feature columns (exclude target)
    all_features = [col for col in train_df.columns if col != target_column_name]

    # Prepare data
    X_train = train_df.select(all_features).to_numpy().astype(np.float32)
    y_train = (
        train_df.select(target_column_name).to_numpy().flatten().astype(np.float32)
    )
    X_val = val_df.select(all_features).to_numpy().astype(np.float32)
    y_val = val_df.select(target_column_name).to_numpy().flatten().astype(np.float32)
    X_test = test_df.select(all_features).to_numpy().astype(np.float32)
    y_test = test_df.select(target_column_name).to_numpy().flatten().astype(np.float32)

    # Standardise features for neural network
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test_scaled)

    print("Feature distribution for Neural Network:")
    print(f"  Input features: {len(all_features)} features")
    print(f"  Feature names: {all_features}")

    # Define neural network architecture
    class MLP(nn.Module):
        def __init__(self, input_size, hidden_layers, hidden_neurons):
            super(MLP, self).__init__()

            layers = []

            # Input layer
            if hidden_layers > 0:
                layers.append(nn.Linear(input_size, hidden_neurons))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))

                # Hidden layers
                for _ in range(hidden_layers - 1):
                    layers.append(nn.Linear(hidden_neurons, hidden_neurons))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(0.2))

                # Output layer
                layers.append(nn.Linear(hidden_neurons, 1))
            else:
                # Direct input to output (logistic regression)
                layers.append(nn.Linear(input_size, 1))

            layers.append(nn.Sigmoid())
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    # Training function
    def train_model(
        model, X_train, y_train, X_val, y_val, epochs=100, lr=0.001, batch_size=32
    ):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        best_val_f1 = 0.0
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val).squeeze()
                val_predictions = (val_outputs > 0.5).float()
                val_f1 = f1_score(y_val.numpy(), val_predictions.numpy())

            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Load best model
        model.load_state_dict(best_model_state)
        return model, best_val_f1

    # Hyperparameter grid search
    hidden_layers_options = [0, 1, 2, 3]  # 0 means logistic regression
    hidden_neurons_options = [16, 32, 64, 128]
    learning_rates = [0.001, 0.01]

    best_params = {"hidden_layers": 1, "hidden_neurons": 32, "lr": 0.001}
    best_f1_val = 0.0

    print("Starting hyperparameter tuning for Neural Network...")
    print(
        f"Testing {len(hidden_layers_options)} layer configs, {len(hidden_neurons_options)} neuron counts, {len(learning_rates)} learning rates..."
    )

    total_combinations = (
        len(hidden_layers_options) * len(hidden_neurons_options) * len(learning_rates)
    )
    current_combination = 0

    for hidden_layers in hidden_layers_options:
        for hidden_neurons in hidden_neurons_options:
            for lr in learning_rates:
                current_combination += 1

                # Skip hidden_neurons when hidden_layers is 0 (logistic regression)
                if hidden_layers == 0 and hidden_neurons != hidden_neurons_options[0]:
                    continue

                # Create and train model
                model = MLP(len(all_features), hidden_layers, hidden_neurons)
                trained_model, val_f1 = train_model(
                    model,
                    X_train_tensor,
                    y_train_tensor,
                    X_val_tensor,
                    y_val_tensor,
                    epochs=200,
                    lr=lr,
                    batch_size=32,
                )

                if current_combination % 5 == 0 or val_f1 > best_f1_val:
                    print(
                        f"  Progress: {current_combination}/{total_combinations} - "
                        f"layers={hidden_layers}, neurons={hidden_neurons}, lr={lr:.3f}, "
                        f"F1={val_f1:.4f}"
                    )

                if val_f1 > best_f1_val:
                    best_f1_val = val_f1
                    best_params["hidden_layers"] = hidden_layers
                    best_params["hidden_neurons"] = hidden_neurons
                    best_params["lr"] = lr

    print("Best parameters found:")
    print(f"  Hidden layers: {best_params['hidden_layers']}")
    print(f"  Hidden neurons: {best_params['hidden_neurons']}")
    print(f"  Learning rate: {best_params['lr']:.3f}")
    print(f"  Validation F1 Score: {best_f1_val:.4f}")

    # Train final model with best parameters
    final_model = MLP(
        len(all_features), best_params["hidden_layers"], best_params["hidden_neurons"]
    )

    # Time the training phase
    start_train_time = perf_counter_ns()
    final_model, _ = train_model(
        final_model,
        X_train_tensor,
        y_train_tensor,
        X_val_tensor,
        y_val_tensor,
        epochs=200,
        lr=best_params["lr"],
        batch_size=32,
    )
    end_train_time = perf_counter_ns()
    training_time = (end_train_time - start_train_time) / 1e9  # Convert to seconds

    # Time the validation prediction
    start_val_time = perf_counter_ns()
    final_model.eval()
    with torch.no_grad():
        val_outputs = final_model(X_val_tensor).squeeze()
        y_val_pred_final = (val_outputs > 0.5).float().numpy()
    end_val_time = perf_counter_ns()
    val_prediction_time = (end_val_time - start_val_time) / 1e9  # Convert to seconds

    f1_val_final = f1_score(y_val, y_val_pred_final)

    # Time the test prediction
    start_test_time = perf_counter_ns()
    with torch.no_grad():
        test_outputs = final_model(X_test_tensor).squeeze()
        y_test_pred_final = (test_outputs > 0.5).float().numpy()
    end_test_time = perf_counter_ns()
    test_prediction_time = (end_test_time - start_test_time) / 1e9  # Convert to seconds

    f1_test_final = f1_score(y_test, y_test_pred_final)

    # Summary
    print("\n\n ## Summary for Neural Network model ##")
    print(f"Final Neural Network Model - Validation F1 Score: {f1_val_final:.4f}")
    print(f"Final Neural Network Model - Test F1 Score: {f1_test_final:.4f}")
    print(f"Training time:           {training_time:.8f} seconds")
    print(f"Validation prediction:   {val_prediction_time:.8f} seconds")
    print(f"Test prediction:         {test_prediction_time:.8f} seconds")

    # Calculate samples per second
    val_samples_per_sec = len(X_val) / val_prediction_time
    print(f"Samples per second (val): {val_samples_per_sec:.0f} samples/sec")

    test_samples_per_sec = len(X_test) / test_prediction_time
    print(f"Samples per second (test): {test_samples_per_sec:.0f} samples/sec")

    # Calculate total samples per second
    total_samples = len(X_val) + len(X_test)
    total_inference_time = val_prediction_time + test_prediction_time
    total_samples_per_sec = total_samples / total_inference_time
    print(f"Samples per second (total): {total_samples_per_sec:.0f} samples/sec")

    # Confusion Matrix for Test Set
    if INTERNAL_PLOTS:
        cm_nn = confusion_matrix(y_test, y_test_pred_final)
        cm_fig_nn = px.imshow(
            cm_nn,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["No Risk (0)", "Risk (1)"],
            y=["No Risk (0)", "Risk (1)"],
            text_auto=True,
            title="Neural Network Confusion Matrix for Test Set",
            color_continuous_scale=px.colors.sequential.Reds,
        )
        cm_fig_nn.update_layout(xaxis_title="Predicted Label", yaxis_title="True Label")
        cm_fig_nn.show()

    # ROC Curve and AUC for Test Set
    with torch.no_grad():
        y_test_proba = final_model(X_test_tensor).squeeze().numpy()
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)

    if INTERNAL_PLOTS:
        roc_fig = go.Figure()
        roc_fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"Neural Network ROC curve (AUC = {roc_auc:.2f})",
            )
        )
        roc_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Chance (AUC = 0.5)",
                line=dict(dash="dash"),
            )
        )

        roc_fig.update_layout(
            title="Neural Network Receiver Operating Characteristic (ROC) Curve for Test Set",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=0.7, y=0.1),
        )
        roc_fig.show()

    # Return ROC data for combined plotting
    return fpr, tpr, roc_auc, "Neural Network"


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Control plot display
    INTERNAL_PLOTS = False
    COMPARISON_PLOTS = True

    # Set seed for reproducibility
    set_seeds(42)

    # Load dataset
    lf = pl.scan_csv("heart 1.csv")

    # Preprocess data
    processed_df: pl.DataFrame = preprocess_data(lf)

    # Split data
    train_df, val_df, test_df = split_data(
        processed_df.clone(), test_size=0.1, validation_size=0.1
    )

    print("\n-------------------------KNN MODEL EVALUATION--------------------------")
    # KNN Workflow - with standard scaling for numerical features
    knn_fpr, knn_tpr, knn_auc, knn_label = knn_model(
        train_df.clone(), val_df.clone(), test_df.clone()
    )

    print(
        "\n-------------------------MIXED NAIVE BAYES MODEL EVALUATION--------------------------"
    )
    # Mixed Naive Bayes Workflow - no additional preprocessing needed
    nb_fpr, nb_tpr, nb_auc, nb_label = mixed_naive_bayes_model(
        train_df.clone(), val_df.clone(), test_df.clone()
    )

    print(
        "\n-------------------------NEURAL NETWORK MODEL EVALUATION--------------------------"
    )
    # Neural Network Workflow - with feature standardisation
    nn_fpr, nn_tpr, nn_auc, nn_label = neural_network_model(
        train_df.clone(), val_df.clone(), test_df.clone()
    )

    if COMPARISON_PLOTS:
        # Create combined ROC curve plot
        combined_roc_fig = go.Figure()

        # Add KNN ROC curve
        combined_roc_fig.add_trace(
            go.Scatter(
                x=knn_fpr,
                y=knn_tpr,
                mode="lines",
                name=f"{knn_label} (AUC = {knn_auc:.3f})",
                line=dict(width=2),
            )
        )

        # Add Mixed Naive Bayes ROC curve
        combined_roc_fig.add_trace(
            go.Scatter(
                x=nb_fpr,
                y=nb_tpr,
                mode="lines",
                name=f"{nb_label} (AUC = {nb_auc:.3f})",
                line=dict(width=2),
            )
        )

        # Add Neural Network ROC curve
        combined_roc_fig.add_trace(
            go.Scatter(
                x=nn_fpr,
                y=nn_tpr,
                mode="lines",
                name=f"{nn_label} (AUC = {nn_auc:.3f})",
                line=dict(width=2),
            )
        )

        # Add random chance line
        combined_roc_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Chance (AUC = 0.500)",
                line=dict(dash="dash", color="gray", width=1),
            )
        )

        combined_roc_fig.update_layout(
            title="Model Comparison: ROC Curves for Heart Disease Prediction",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=0.6, y=0.15),
            font=dict(size=12),
        )

        combined_roc_fig.show()

    # Print summary comparison
    print("\nModel Performance Summary (AUC):")
    print(f"  {knn_label}: {knn_auc:.4f}")
    print(f"  {nb_label}: {nb_auc:.4f}")
    print(f"  {nn_label}: {nn_auc:.4f}")

    # Determine best model
    models = [(knn_auc, knn_label), (nb_auc, nb_label), (nn_auc, nn_label)]
    best_auc, best_model = max(models)
    print(f"\nBest performing model: {best_model} (AUC = {best_auc:.4f})")
