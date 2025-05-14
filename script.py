
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import numpy as np
import pandas as pd
import argparse
import sklearn

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, 'model.joblib'))
    return clf

if __name__ == "__main__":
    print("[INFO] Extracting arguments")
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=0)

    # Use environment variables with proper fallbacks
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test"))
    parser.add_argument("--train-file", type=str, default="train-V-1.csv")
    parser.add_argument("--test-file", type=str, default="test-V-1.csv")

    args = parser.parse_args()

    print("SKLearn Version: ", sklearn.__version__)

    # List directory contents to debug
    print(f"Contents of train directory: {os.listdir(args.train)}")
    print(f"Contents of test directory: {os.listdir(args.test)}")

    # Read training data
    train_path = os.path.join(args.train, args.train_file)
    test_path = os.path.join(args.test, args.test_file)

    print("[INFO] Reading training data")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Training data shape: {train_df.shape}")
    print(f"Testing data shape: {test_df.shape}")

    # Extract features and labels
    features = list(train_df.columns)
    label = features.pop(-1)  # Assumes label is last column

    print("Column order:")
    print(features)
    print()

    print("Label column is: ", label)
    print()

    X_train = train_df[features]
    y_train = train_df[label]

    # Extract test features
    # Check if test data has the label column
    has_label_in_test = label in test_df.columns

    if has_label_in_test:
        X_test = test_df[features]
        y_test = test_df[label]
    else:
        print(f"Warning: Test data does not contain the label column '{label}'")
        print("Will only generate predictions, no evaluation metrics")
        # Use all columns in test data as features
        X_test = test_df
        y_test = None

    print("[INFO] Training RandomForest Model")
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state, verbose=0)
    model.fit(X_train, y_train)

    # Create model directory if it doesn't exist
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)

    # Save the trained model
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Model evaluation (only if we have test labels)
    if y_test is not None:
        y_pred_test = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred_test)
        test_rep = classification_report(y_test, y_pred_test)

        print("\n----METRICS RESULT FOR TESTING DATA----\n")
        print(f"[Testing] Model Accuracy is: {test_acc}")
        print("[Testing] Testing Report: ")
        print(test_rep)
    else:
        # Just generate predictions
        y_pred_test = model.predict(X_test)
        print("\n----PREDICTIONS FOR TESTING DATA----\n")
        print("First 10 predictions:", y_pred_test[:10])

        # Optionally save predictions to a file
        pred_df = pd.DataFrame({'predicted_price_range': y_pred_test})
        pred_path = os.path.join(args.model_dir, "predictions.csv")
        pred_df.to_csv(pred_path, index=False)
        print(f"Predictions saved to {pred_path}")
