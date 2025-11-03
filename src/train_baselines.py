"""
train_baselines.py
- Trains classical ML models for classification and regression
- Logs params, metrics, and artifacts to MLflow
- Saves trained models to models/
"""
import argparse
import joblib
import os
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error
from features import load_vectorizer, transform_csv
from utils import ensure_dir, set_seed

set_seed(42)
MLFLOW_EXPERIMENT = "imdb_midpoint"

def eval_classif(model, X, y):
    preds = model.predict(X)
    probs = None
    try:
        probs = model.predict_proba(X)[:,1]
    except Exception:
        probs = None
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, pos_label="positive")
    roc = roc_auc_score([1 if lab=="positive" else 0 for lab in y], probs) if probs is not None else np.nan
    return {"accuracy": float(acc), "f1": float(f1), "roc_auc": float(roc)}

def eval_reg(model, X, y):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = mean_squared_error(y, preds, squared=False)
    return {"mae": float(mae), "rmse": float(rmse)}

def train_classification(X_train, y_train, X_val, y_val, models_dir, vectorizer_path):
    ensure_dir(models_dir)
    # Models to train: Logistic Regression, MultinomialNB, Decision Tree
    models = {
        "logreg": LogisticRegression(max_iter=200, solver="liblinear"),
        "mnb": MultinomialNB(),
        "dt": DecisionTreeClassifier(random_state=42)
    }
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    results = {}
    for name, model in models.items():
        with mlflow.start_run(run_name=f"classification_{name}"):
            model.fit(X_train, y_train)
            train_metrics = eval_classif(model, X_train, y_train)
            val_metrics = eval_classif(model, X_val, y_val)
            mlflow.log_params({"model": name})
            mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
            # save model
            path = os.path.join(models_dir, f"classification_{name}.joblib")
            joblib.dump(model, path)
            mlflow.log_artifact(path)
            results[name] = {"model": model, "train": train_metrics, "val": val_metrics, "path": path}
            print(f"Trained {name}: val f1={val_metrics['f1']:.4f}")
    return results

def train_regression(X_train, y_train, X_val, y_val, models_dir):
    ensure_dir(models_dir)
    models = {
        "linear": LinearRegression(),
        "dt_reg": DecisionTreeRegressor(random_state=42)
    }
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    results = {}
    for name, model in models.items():
        with mlflow.start_run(run_name=f"regression_{name}"):
            model.fit(X_train, y_train)
            train_metrics = eval_reg(model, X_train, y_train)
            val_metrics = eval_reg(model, X_val, y_val)
            mlflow.log_params({"model": name})
            mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
            path = os.path.join(models_dir, f"regression_{name}.joblib")
            joblib.dump(model, path)
            mlflow.log_artifact(path)
            results[name] = {"model": model, "train": train_metrics, "val": val_metrics, "path": path}
            print(f"Trained {name}: val mae={val_metrics['mae']:.4f}")
    return results

def load_data_matrices(data_dir, vectorizer_path, text_col="review", class_col="sentiment", reg_col="rating"):
    # Transform CSVs to matrices
    X_train, df_train = transform_csv(f"{data_dir}/train.csv", vectorizer_path, text_col=text_col)
    X_val, df_val = transform_csv(f"{data_dir}/val.csv", vectorizer_path, text_col=text_col)
    X_test, df_test = transform_csv(f"{data_dir}/test.csv", vectorizer_path, text_col=text_col)
    y_train_cl = df_train[class_col].astype(str).tolist()
    y_val_cl = df_val[class_col].astype(str).tolist()
    y_test_cl = df_test[class_col].astype(str).tolist()
    # regression targets may be missing (NaN)
    try:
        y_train_reg = df_train[reg_col].astype(float).to_numpy()
        y_val_reg = df_val[reg_col].astype(float).to_numpy()
        y_test_reg = df_test[reg_col].astype(float).to_numpy()
    except Exception:
        # fallback to zeros if no reg target
        y_train_reg = np.zeros(len(df_train))
        y_val_reg = np.zeros(len(df_val))
        y_test_reg = np.zeros(len(df_test))
    return (X_train, y_train_cl, y_train_reg), (X_val, y_val_cl, y_val_reg), (X_test, y_test_cl, y_test_reg)

def main(args):
    set_seed(args.seed)
    ensure_dir(args.models_dir)
    # ensure vectorizer exists (fit if not)
    if not os.path.exists(args.vectorizer_path):
        from features import fit_vectorizer
        fit_vectorizer(args.data_dir + "/train.csv", args.vectorizer_path, text_col=args.text_col)
    (X_train, y_train_cl, y_train_reg), (X_val, y_val_cl, y_val_reg), (X_test, y_test_cl, y_test_reg) = load_data_matrices(
        args.data_dir, args.vectorizer_path, text_col=args.text_col, class_col=args.class_col, reg_col=args.reg_col)
    # Train classifiers
    cls_results = train_classification(X_train, y_train_cl, X_val, y_val_cl, args.models_dir, args.vectorizer_path)
    # Train regressors (only if numeric targets exist)
    reg_results = None
    # check for non-trivial regression targets
    if not np.all(np.isnan(y_train_reg)) and np.nanstd(y_train_reg) > 0:
        reg_results = train_regression(X_train, y_train_reg, X_val, y_val_reg, args.models_dir)
    else:
        print("Regression target not found or constant — skipping regression training.")
    print("Training completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Processed CSV directory with train/val/test")
    parser.add_argument("--vectorizer_path", type=str, default="models/tfidf_vectorizer.joblib")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--text_col", type=str, default="review")
    parser.add_argument("--class_col", type=str, default="sentiment")
    parser.add_argument("--reg_col", type=str, default="rating")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
