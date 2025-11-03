"""
AI Tutor Response Evaluation - Data Processing Utilities
Repository: https://github.com/tituatgithub/INDO_ML

Contributors:
- @Kweenbee187
- @tituatgithub

This module handles data loading, preprocessing, and augmentation
for the AI tutor response evaluation task.
"""

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


def load_training_data(filepath):
    """
    Load and flatten training JSON data
    
    Args:
        filepath (str): Path to training JSON file
    
    Returns:
        pd.DataFrame: Flattened DataFrame with columns:
            - conversation_id
            - conversation_history
            - tutor
            - response
            - Mistake_Identification
    
    Example:
        >>> df = load_training_data("data/trainset.json")
        >>> print(df.shape)
        (2476, 5)
    """
    print(f"Loading training data from: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    
    rows = []
    for item in train_data:
        conv_id = item.get("conversation_id", "")
        history = item.get("conversation_history", "")
        tutor_responses = item.get("tutor_responses", {})
        
        for tutor_name, info in tutor_responses.items():
            response_text = info.get("response", "")
            annotation = info.get("annotation", {})
            mi = annotation.get("Mistake_Identification", None)
            
            # Only include samples with labels
            if mi is not None:
                rows.append({
                    "conversation_id": conv_id,
                    "conversation_history": history,
                    "tutor": tutor_name,
                    "response": response_text,
                    "Mistake_Identification": mi
                })
    
    df = pd.DataFrame(rows)
    
    print(f"Training dataset shape: {df.shape}")
    print("\nClass distribution:")
    print(df["Mistake_Identification"].value_counts())
    print()
    
    return df


def load_test_data(filepath):
    """
    Load test JSON data (without labels)
    
    Args:
        filepath (str): Path to test JSON file
    
    Returns:
        pd.DataFrame: DataFrame with columns:
            - conversation_id
            - conversation_history
            - tutor
            - response
    
    Example:
        >>> test_df = load_test_data("data/testset.json")
        >>> print(test_df.shape)
        (1214, 4)
    """
    print(f"Loading test data from: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    rows = []
    for item in test_data:
        conv_id = item.get("conversation_id", "")
        history = item.get("conversation_history", "")
        tutor_responses = item.get("tutor_responses", {})
        
        for tutor_name, info in tutor_responses.items():
            response_text = info.get("response", "")
            rows.append({
                "conversation_id": conv_id,
                "conversation_history": history,
                "tutor": tutor_name,
                "response": response_text
            })
    
    df = pd.DataFrame(rows)
    print(f"Test dataset shape: {df.shape}\n")
    
    return df


def concat_text(row, max_history_turns=3):
    """
    Concatenate conversation history with tutor response
    
    Uses [SEP] tokens as delimiters between conversation turns
    and keeps only the last N turns of conversation history.
    
    Args:
        row (pd.Series): DataFrame row with 'conversation_history' and 'response'
        max_history_turns (int): Maximum number of conversation turns to include
    
    Returns:
        str: Concatenated text with format:
             "turn1 [SEP] turn2 [SEP] turn3 [SEP] response"
    
    Example:
        >>> text = concat_text(df.iloc[0], max_history_turns=3)
        >>> print(text)
        "Student: I got x=5 [SEP] Tutor: That's correct [SEP] Response text"
    """
    # Split conversation history into lines
    history_lines = row['conversation_history'].strip().split("\n")
    
    # Keep only last N turns
    if len(history_lines) > max_history_turns:
        history_lines = history_lines[-max_history_turns:]
    
    # Join with separator, filtering empty lines
    context = " [SEP] ".join([line.strip() for line in history_lines if line.strip()])
    
    # Concatenate with response
    return context + " [SEP] " + row['response'].strip()


def minimal_augment(df, multiplier=2):
    """
    Apply minimal but effective augmentation for minority class
    
    Adds simple linguistic variations to "To some extent" samples
    to balance the dataset without over-complicating.
    
    Args:
        df (pd.DataFrame): Training DataFrame
        multiplier (int): How many augmented versions per sample (default: 2)
    
    Returns:
        pd.DataFrame: Augmented DataFrame
    
    Example:
        >>> df_aug = minimal_augment(df, multiplier=2)
        >>> print(df_aug['Mistake_Identification'].value_counts())
    """
    rows = df.to_dict(orient="records")
    partial = df[df["Mistake_Identification"] == "To some extent"]
    
    print(f"\nOriginal 'To some extent' samples: {len(partial)}")
    
    # Add augmented versions
    for _, row in partial.iterrows():
        if multiplier >= 1:
            # Version 1: Add uncertainty marker
            aug1 = row.copy()
            aug1['response'] = row['response'] + " I think."
            rows.append(aug1.to_dict())
        
        if multiplier >= 2:
            # Version 2: Add qualifier
            aug2 = row.copy()
            aug2['response'] = "Somewhat, " + row['response'].lower()
            rows.append(aug2.to_dict())
        
        if multiplier >= 3:
            # Version 3: Add hedging phrase
            aug3 = row.copy()
            aug3['response'] = row['response'] + " Perhaps."
            rows.append(aug3.to_dict())
    
    result_df = pd.DataFrame(rows)
    
    print(f"After augmentation (multiplier={multiplier}):")
    print(result_df['Mistake_Identification'].value_counts())
    print()
    
    return result_df


def encode_labels(df, label_column='Mistake_Identification'):
    """
    Encode text labels to integers
    
    Args:
        df (pd.DataFrame): DataFrame with label column
        label_column (str): Name of column containing labels
    
    Returns:
        tuple: (df with encoded labels, LabelEncoder instance, unique classes)
    
    Example:
        >>> df, le, classes = encode_labels(df)
        >>> print({i: le.inverse_transform([i])[0] for i in classes})
        {0: 'No', 1: 'To some extent', 2: 'Yes'}
    """
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df[label_column])
    classes_unique = np.unique(df["label_enc"])
    
    print("Label mapping:")
    for i in classes_unique:
        print(f"  {i}: {le.inverse_transform([i])[0]}")
    print()
    
    return df, le, classes_unique


def compute_class_weights(labels, classes_unique, minority_boost=1.5, clip_range=(0.8, 3.0)):
    """
    Compute balanced class weights with optional minority boost
    
    Args:
        labels (np.ndarray): Array of encoded labels
        classes_unique (np.ndarray): Array of unique class indices
        minority_boost (float): Additional multiplier for minority class
        clip_range (tuple): (min, max) range to clip weights
    
    Returns:
        tuple: (class_weights array, class_weights tensor)
    
    Example:
        >>> weights, weights_tensor = compute_class_weights(labels, classes, minority_boost=1.5)
        >>> print(dict(zip(classes, weights)))
        {0: 2.54, 1: 2.70, 2: 0.80}
    """
    import torch
    
    # Compute balanced weights
    class_weights = compute_class_weight(
        "balanced",
        classes=classes_unique,
        y=labels
    )
    
    # Apply minority boost to "To some extent" class (index 1)
    if 1 in classes_unique:
        class_weights[1] *= minority_boost
    
    # Clip weights to reasonable range
    class_weights = np.clip(class_weights, clip_range[0], clip_range[1])
    
    # Convert to tensor
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    
    print("Class weights:")
    for i, weight in zip(classes_unique, class_weights):
        print(f"  Class {i}: {weight:.4f}")
    print()
    
    return class_weights, class_weights_tensor


def preprocess_data(train_df, max_history_turns=3):
    """
    Apply text preprocessing to entire DataFrame
    
    Args:
        train_df (pd.DataFrame): Training DataFrame
        max_history_turns (int): Maximum conversation turns to include
    
    Returns:
        list: List of preprocessed text strings
    
    Example:
        >>> texts = preprocess_data(df, max_history_turns=3)
        >>> print(len(texts))
        2824
    """
    texts = train_df.apply(concat_text, axis=1, max_history_turns=max_history_turns).tolist()
    print(f"Preprocessed {len(texts)} text samples\n")
    return texts


def save_predictions(predictions_dict, output_path):
    """
    Save predictions to JSON file
    
    Args:
        predictions_dict (dict): Dictionary of predictions by conversation_id
        output_path (str): Path to save JSON file
    
    Example:
        >>> save_predictions(preds, "outputs/predictions.json")
        Saved predictions for 150 conversations to outputs/predictions.json
    """
    pred_list = list(predictions_dict.values())
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pred_list, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved predictions for {len(pred_list)} conversations to {output_path}")


def create_prediction_json(test_df, predicted_labels):
    """
    Create prediction dictionary in required format
    
    Args:
        test_df (pd.DataFrame): Test DataFrame
        predicted_labels (np.ndarray): Array of predicted label strings
    
    Returns:
        dict: Predictions organized by conversation_id
    
    Example:
        >>> preds = create_prediction_json(test_df, final_labels)
        >>> print(len(preds))
        150
    """
    predictions = {}
    
    for idx, row in test_df.iterrows():
        conv_id = row['conversation_id']
        tutor = row['tutor']
        predicted_label = predicted_labels[idx]
        
        # Initialize conversation if not exists
        if conv_id not in predictions:
            predictions[conv_id] = {
                "conversation_id": conv_id,
                "conversation_history": row['conversation_history'],
                "tutor_responses": {}
            }
        
        # Add tutor response with prediction
        predictions[conv_id]["tutor_responses"][tutor] = {
            "response": row['response'],
            "annotation": {"Mistake_Identification": predicted_label}
        }
    
    print(f"\nGenerated predictions for {len(predictions)} conversations")
    
    return predictions


def get_class_distribution(labels, label_encoder):
    """
    Get detailed class distribution statistics
    
    Args:
        labels (np.ndarray): Array of encoded labels
        label_encoder: Sklearn LabelEncoder
    
    Returns:
        dict: Class distribution with counts and percentages
    
    Example:
        >>> dist = get_class_distribution(labels, le)
        >>> print(dist['Yes']['count'])
        1932
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    distribution = {}
    for class_idx, count in zip(unique, counts):
        class_name = label_encoder.inverse_transform([class_idx])[0]
        distribution[class_name] = {
            'count': int(count),
            'percentage': float(count / total * 100)
        }
    
    return distribution


def validate_data(df, required_columns=None):
    """
    Validate DataFrame has required columns and no missing values
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
    
    Raises:
        ValueError: If validation fails
    
    Example:
        >>> validate_data(df, ['conversation_id', 'response'])
    """
    if required_columns is None:
        required_columns = ['conversation_id', 'conversation_history', 'tutor', 'response']
    
    # Check for required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for missing values
    for col in required_columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            print(f"Warning: {null_count} missing values in column '{col}'")
    
    print("✓ Data validation passed\n")


# Data statistics functions
def print_data_statistics(df, label_column='Mistake_Identification'):
    """
    Print comprehensive data statistics
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
        label_column (str): Name of label column
    """
    print(f"\n{'='*60}")
    print("DATA STATISTICS")
    print(f"{'='*60}")
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Total samples: {len(df)}")
    print(f"Unique conversations: {df['conversation_id'].nunique()}")
    print(f"Unique tutors: {df['tutor'].nunique()}")
    
    if label_column in df.columns:
        print(f"\nClass distribution:")
        class_dist = df[label_column].value_counts()
        for label, count in class_dist.items():
            percentage = count / len(df) * 100
            print(f"  {label:<20} {count:>6} ({percentage:>5.2f}%)")
    
    # Response length statistics
    df['response_length'] = df['response'].str.len()
    print(f"\nResponse length statistics:")
    print(f"  Mean: {df['response_length'].mean():.1f} characters")
    print(f"  Median: {df['response_length'].median():.1f} characters")
    print(f"  Min: {df['response_length'].min()} characters")
    print(f"  Max: {df['response_length'].max()} characters")
    print(f"{'='*60}\n")
