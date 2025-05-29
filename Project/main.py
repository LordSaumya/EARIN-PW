import polars as pl
import numpy as np
from imblearn.over_sampling import SMOTE
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc

# Global consts for feature names and target column
TARGET_COLUMN = "Heart Attack Risk"
ORIGINAL_BINARY_FEATURES = [
    "Diabetes", "Family History", "Smoking", "Obesity",
    "Alcohol Consumption", "Previous Heart Problems", "Medication Use"
]
DISCRETE_NUMERICAL_FEATURES_INFO = {
    "Physical Activity Days Per Week": (0, 7),
    "Sleep Hours Per Day": (1, 24),
    "Stress Level": (1, 10),
}
LABEL_ENCODED_COLUMN_NAMES = ["Country_encoded", "Continent_encoded"]
ONE_HOT_PREFIXES = ["Sex_", "Diet_", "Hemisphere_"]


def preprocess_data(df: pl.LazyFrame) -> pl.DataFrame:
    # Separate blood pressure into systolic and diastolic and drop original column)
    df_engineered = df.with_columns([
        pl.col("Blood Pressure").str.split("/").list.get(0).cast(pl.Int32).alias("Systolic_BP"),
        pl.col("Blood Pressure").str.split("/").list.get(1).cast(pl.Int32).alias("Diastolic_BP")
    ]).drop("Blood Pressure")

    # Drop Patient ID if it exists, as it's not a feature
    # Using select all except "Patient ID". This won't error if "Patient ID" doesn't exist.
    df_engineered = df_engineered.select(pl.all().exclude("Patient ID"))
    
    numerical_features: list[str] = [
        "Age", "Cholesterol", "Heart Rate", "Systolic_BP", "Diastolic_BP",
        "Exercise Hours Per Week", "Stress Level", "Sedentary Hours Per Day",
        "Income", "BMI", "Triglycerides", "Physical Activity Days Per Week",
        "Sleep Hours Per Day"
    ]

    # _binary_features list is now global: ORIGINAL_BINARY_FEATURES
    # _categorical_features list is used for deciding encoding strategy
    _categorical_features: list[str] = ["Sex", "Diet", "Country", "Continent", "Hemisphere"]

    # target_column is global: TARGET_COLUMN

    # Start with engineered dataframe
    df_encoded: pl.LazyFrame = df_engineered

    # One-hot encoding for low cardinality features
    low_cardinality: list[str] = ["Sex", "Diet", "Hemisphere"]
    
    # Get schema once to avoid performance warnings
    schema_names = df_encoded.collect_schema().names()

    for feature in low_cardinality:
        if feature in schema_names:
            # Get unique values for this feature
            unique_values: list[str] = df_encoded.select(pl.col(feature).unique()).collect().get_column(feature).to_list()

            # Add all dummy columns and drop original
            dummy_expressions: list[pl.Expr] = []
            for value in unique_values:
                dummy_expressions.append(
                    (pl.col(feature) == value).cast(pl.Int32).alias(f"{feature}_{value}")
                )
            df_encoded: pl.LazyFrame = df_encoded.with_columns(dummy_expressions).drop(feature)

    # Label encoding for high cardinality features
    high_cardinality: list[str] = ["Country", "Continent"]
    
    # Update schema names after one-hot encoding changes
    schema_names = df_encoded.collect_schema().names()
    
    for feature in high_cardinality:
        if feature in schema_names:
            # Get unique values to create mapping
            unique_values: list[str] = df_encoded.select(pl.col(feature).unique()).collect().get_column(feature).to_list()
            
            # Create mapping dictionary
            mapping = {value: idx for idx, value in enumerate(sorted(unique_values))}
            
            # Apply mapping and drop original column
            df_encoded = df_encoded.with_columns(
                pl.col(feature).map_elements(lambda x: mapping.get(x, -1), return_dtype=pl.Int32).alias(f"{feature}_encoded")
            ).drop(feature)
        
    # Outlier Detection and Removal
    outlier_conditions = []
    
    # Update schema names after label encoding changes
    schema_names = df_encoded.collect_schema().names()
    
    for feature in numerical_features:
        if feature in schema_names:
            # Calculate quartiles
            quartiles = df_encoded.select([
                pl.col(feature).quantile(0.25).alias("Q1"),
                pl.col(feature).quantile(0.75).alias("Q3")
            ]).collect()
            
            Q1 = quartiles["Q1"].item()
            Q3 = quartiles["Q3"].item()
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Remove outliers
            outlier_conditions.append(
                (pl.col(feature) >= lower_bound) & (pl.col(feature) <= upper_bound)
            )
    
    # Apply all outlier removal conditions
    if outlier_conditions:
        combined_condition = outlier_conditions[0]
        for condition in outlier_conditions[1:]:
            combined_condition = combined_condition & condition
        
        df_clean = df_encoded.filter(combined_condition)
    else:
        df_clean = df_encoded
        
    return df_clean.collect()

def split_data(df: pl.DataFrame, test_size: float, validation_size: float) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
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

def apply_smote_and_post_corrections(
    df: pl.DataFrame, 
    target_col_name: str
) -> pl.DataFrame:
    
    feature_columns = [col for col in df.columns if col != target_col_name]
    
    X = df.select(feature_columns).to_numpy()
    y = df.select(target_col_name).to_numpy().flatten()
        
    _ , counts = np.unique(y, return_counts=True)
    min_class_count = counts.min() if len(counts) > 0 else 0
    
    smote_k_neighbors = min(5, min_class_count - 1) 
    # Ensure k_neighbors is at least 1
    smote_k_neighbors = max(1, smote_k_neighbors)

    smote = SMOTE(random_state=42, k_neighbors=smote_k_neighbors)
    X_balanced, y_balanced = smote.fit_resample(X, y)
        
    onehot_features = [col for col in feature_columns if any(col.startswith(prefix) for prefix in ONE_HOT_PREFIXES)]
    
    # Ensure categorical features are binary or within specified ranges
    for i, col_name in enumerate(feature_columns):
        if col_name in ORIGINAL_BINARY_FEATURES:
            X_balanced[:, i] = np.clip(np.round(X_balanced[:, i]), 0, 1).astype(int)
        elif col_name in onehot_features:
            X_balanced[:, i] = np.clip(np.round(X_balanced[:, i]), 0, 1).astype(int)
        elif col_name in DISCRETE_NUMERICAL_FEATURES_INFO:
            min_val, max_val = DISCRETE_NUMERICAL_FEATURES_INFO[col_name]
            X_balanced[:, i] = np.clip(np.round(X_balanced[:, i]), min_val, max_val).astype(int)
        elif col_name in LABEL_ENCODED_COLUMN_NAMES:
            X_balanced[:, i] = np.maximum(0, np.round(X_balanced[:, i])).astype(int)
    
    # Correct one-hot encoded groups
    for prefix_idx, one_hot_prefix in enumerate(ONE_HOT_PREFIXES):
        group_cols_indices = [i for i, col in enumerate(feature_columns) if col.startswith(one_hot_prefix)]
        if not group_cols_indices:
            continue

        group_data = X_balanced[:, group_cols_indices]
        # Ensure each row in the group sums to 1
        # If a row sums to 0, randomly pick one column in the group to set to 1
        # If a row sums to >1, pick the column with the highest value (or randomly if ties)
        for row_idx in range(X_balanced.shape[0]):
            row_sum = group_data[row_idx, :].sum()
            if row_sum != 1:
                X_balanced[row_idx, group_cols_indices] = 0
                if len(group_cols_indices) > 0:
                    if row_sum == 0 or True:
                        # Randomly pick one column in the group to set to 1
                        chosen_col_for_row = np.random.choice(group_cols_indices)
                        X_balanced[row_idx, chosen_col_for_row] = 1


    # Return balanced DataFrame
    balanced_data_dict = {}
    for i, col_name in enumerate(feature_columns):
        balanced_data_dict[col_name] = X_balanced[:, i]
    balanced_data_dict[target_col_name] = y_balanced
    
    return pl.DataFrame(balanced_data_dict)


# Function for KNN-specific preprocessing
def preprocess_for_knn(
    df: pl.DataFrame,
    existing_scaler: StandardScaler = None,
    target_column_name: str = TARGET_COLUMN # Use global constant
) -> Tuple[pl.DataFrame, StandardScaler]:

    # Exclude label encoded features and target column from scaling
    columns_to_exclude_list = list(LABEL_ENCODED_COLUMN_NAMES) 

    if target_column_name in df.columns and target_column_name not in columns_to_exclude_list:
        columns_to_exclude_list.append(target_column_name)

    features_to_scale = [col for col in df.columns if col not in columns_to_exclude_list]

    scaler = StandardScaler() if existing_scaler is None else existing_scaler

    if not features_to_scale: # Handle case with no features to scale
        if existing_scaler is None: # fit call
             scaler.fit(np.empty((df.height, 0))) # Fit on empty data
        return df.clone(), scaler # Return original df if no features to scale

    data_to_scale_np = df.select(features_to_scale).to_numpy()
    if existing_scaler is None: 
        scaled_data_np = scaler.fit_transform(data_to_scale_np)
    else: 
        scaled_data_np = scaler.transform(data_to_scale_np)
    scaled_features_df = pl.DataFrame(data=scaled_data_np, schema=features_to_scale)

    # Reconstruct DataFrame: scaled features + non-scaled (target and explicitly excluded)
    output_df_parts = [scaled_features_df]
    columns_not_scaled_but_kept = [col for col in columns_to_exclude_list if col in df.columns]
    
    if columns_not_scaled_but_kept:
        output_df_parts.append(df.select(columns_not_scaled_but_kept))
        
    df_output: pl.DataFrame = pl.concat(output_df_parts, how="horizontal")
    
    # Ensure original column order as much as possible
    original_order = [col for col in df.columns if col in df_output.columns]
    df_output = df_output.select(original_order)

    return df_output, scaler

def knn_model(train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: pl.DataFrame):
    # Preprocess data for KNN
    scaled_train_df_knn, knn_scaler = preprocess_for_knn(train_df.clone())
    scaled_val_df_knn, _ = preprocess_for_knn(val_df.clone(), existing_scaler=knn_scaler)
    scaled_test_df_knn, _ = preprocess_for_knn(test_df.clone(), existing_scaler=knn_scaler)
    
    features_for_model = [
        col for col in scaled_train_df_knn.columns 
        if col != TARGET_COLUMN and col not in LABEL_ENCODED_COLUMN_NAMES
    ]

    X_train = scaled_train_df_knn.select(features_for_model).to_numpy()
    y_train = scaled_train_df_knn.select(TARGET_COLUMN).to_numpy().flatten()
    X_val = scaled_val_df_knn.select(features_for_model).to_numpy()
    y_val = scaled_val_df_knn.select(TARGET_COLUMN).to_numpy().flatten()
    X_test = scaled_test_df_knn.select(features_for_model).to_numpy()
    y_test = scaled_test_df_knn.select(TARGET_COLUMN).to_numpy().flatten()

    # Hyperparameter tuning for n_neighbors using F1 score on validation set
    k_values = list(range(1, 101))  # Testing k from 1 to 100
    best_k = k_values[0]
    best_f1_val = 0.0
    validation_f1_scores = [] # Store F1 scores for plotting

    for k in k_values:
        knn_temp_classifier = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn_temp_classifier.fit(X_train, y_train)
        y_val_pred_temp = knn_temp_classifier.predict(X_val)
        f1_val_temp = f1_score(y_val, y_val_pred_temp)
        validation_f1_scores.append(f1_val_temp)
        print(f"  k={k}: Validation F1 Score = {f1_val_temp:.4f}")
        if f1_val_temp > best_f1_val:
            best_f1_val = f1_val_temp
            best_k = k

    print(f"Best k found: {best_k} with Validation F1 Score: {best_f1_val:.4f}")

    # Plot k_values vs validation_f1_scores
    fig = px.line(x=k_values, y=validation_f1_scores, labels={'x':'k (Number of Neighbors)', 'y':'Validation F1 Score'}, title='KNN Hyperparameter Tuning: k vs. Validation F1 Score')
    fig.show()

    # Train final KNN classifier with the best k
    final_knn_classifier = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
    final_knn_classifier.fit(X_train, y_train)

    # Evaluate final model on validation and test sets
    y_val_pred_final = final_knn_classifier.predict(X_val)
    f1_val_final = f1_score(y_val, y_val_pred_final)
    print(f"Final KNN Model - Validation F1 Score (k={best_k}): {f1_val_final:.4f}")

    y_test_pred_final = final_knn_classifier.predict(X_test)
    f1_test_final = f1_score(y_test, y_test_pred_final)
    print(f"Final KNN Model - Test F1 Score (k={best_k}): {f1_test_final:.4f}")

    # Confusion Matrix for Test Set
    cm = confusion_matrix(y_test, y_test_pred_final)
    cm_fig = px.imshow(cm,
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['No Risk (0)', 'Risk (1)'],
                       y=['No Risk (0)', 'Risk (1)'],
                       text_auto=True,
                       title=f"Confusion Matrix for Test Set (k={best_k})",
                       color_continuous_scale=px.colors.sequential.Blues)
    cm_fig.update_layout(xaxis_title="Predicted Label", yaxis_title="True Label")
    cm_fig.show()

    # ROC Curve and AUC for Test Set
    y_test_proba = final_knn_classifier.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)

    roc_fig = go.Figure()
    # Plot ROC curve
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.2f})'))
    # Plot random chance line
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Chance (AUC = 0.5)', line=dict(dash='dash')))
    
    roc_fig.update_layout(
        title=f'Receiver Operating Characteristic (ROC) Curve for Test Set (k={best_k})',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend=dict(x=0.7, y=0.1)
    )
    roc_fig.show()

if __name__ == "__main__":
    # Load dataset
    lf = pl.scan_csv("heart_attack_prediction_dataset.csv")
    
    # Preprocess data (encoding, outlier removal, NO SMOTE)
    processed_df: pl.DataFrame = preprocess_data(lf)
    
    # Split data into train, validation, and test sets
    train_df, val_df, test_df = split_data(processed_df, test_size=0.1, validation_size=0.1)

    # Apply SMOTE and post-corrections ONLY to the training set
    print(f"Shape of train_df before SMOTE: {train_df.shape}")
    if TARGET_COLUMN in train_df.columns:
        print(f"Class distribution in train_df before SMOTE:\n{train_df[TARGET_COLUMN].value_counts()}")
    
    train_df_smoted = apply_smote_and_post_corrections(train_df, TARGET_COLUMN)
    
    print(f"Shape of train_df after SMOTE: {train_df_smoted.shape}")
    if TARGET_COLUMN in train_df_smoted.columns:
        print(f"Class distribution in train_df after SMOTE:\n{train_df_smoted[TARGET_COLUMN].value_counts()}")

    # Run KNN model workflow
    knn_model(train_df_smoted, val_df, test_df)