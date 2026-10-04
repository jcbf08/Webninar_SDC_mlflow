"""Piezas compartidas: configuración, datos, pipeline y métricas."""

import logging
import os
import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

# Validación cruzada reproducible: mismos folds en todos los experimentos
CV = StratifiedKFold(CFG["cv_folds"], shuffle=True, random_state=CFG["seed"])

NUM = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CAT = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
       "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
       "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
       "PaperlessBilling", "PaymentMethod"]
FEATURES = NUM + CAT

MODELS = {
    "logreg": (LogisticRegression, {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"}),
    "forest": (RandomForestClassifier, {"n_estimators": 300, "max_depth": 8,
                                        "min_samples_leaf": 5, "class_weight": "balanced",
                                        "n_jobs": -1}),
    "boosting": (HistGradientBoostingClassifier, {"learning_rate": 0.1, "max_iter": 200,
                                                  "max_depth": 6, "class_weight": "balanced"}),
}


def setup_mlflow():
    """Conecta con MLflow. Por defecto: un archivo SQLite local (tracking + registry)."""
    warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
    logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)
    uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    mlflow.set_experiment(CFG["experiment"])


def clean(df):
    """Misma limpieza en entrenamiento y en producción."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    for col in NUM:
        df[col] = df[col].astype("float64")
    return df


def load_data():
    """Devuelve X_train, X_test, y_train, y_test."""
    df = clean(pd.read_csv(ROOT / "data" / "churn.csv"))
    missing = set(FEATURES + ["Churn"]) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {sorted(missing)}")
    y = (df["Churn"] == "Yes").astype(int)
    return train_test_split(df[FEATURES], y, test_size=CFG["test_size"], stratify=y,
                            random_state=CFG["seed"])


def build_pipeline(name, params=None):
    """Preprocesamiento + modelo en un solo objeto: lo que se entrena es lo que se sirve."""
    cls, defaults = MODELS[name]
    prep = ColumnTransformer(
        [("num", StandardScaler(), NUM),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)],
        verbose_feature_names_out=False)
    model = cls(**{**defaults, **(params or {}), "random_state": CFG["seed"]})
    return Pipeline([("preprocess", prep), ("model", model)])


def profit_curve(y_true, proba, business=None):
    """Utilidad de la campaña para cada umbral: a quién contactamos y cuánto ganamos."""
    b = business or CFG["business"]
    y_true = np.asarray(y_true)
    rows = []
    for thr in np.round(np.arange(0.05, 0.96, 0.01), 2):
        contacted = proba >= thr
        tp = int((contacted & (y_true == 1)).sum())
        total = int(contacted.sum())
        profit = tp * b["success_rate"] * b["clv_saved"] - total * b["offer_cost"]
        rows.append({"threshold": float(thr), "contacted": total, "profit": float(profit)})
    return pd.DataFrame(rows)


def evaluate(y_true, proba):
    """Métricas técnicas + métrica de negocio. Devuelve (métricas, curva de utilidad)."""
    curve = profit_curve(y_true, proba)
    best = curve.loc[curve["profit"].idxmax()]
    metrics = {
        "test_auc": roc_auc_score(y_true, proba),
        "test_recall": recall_score(y_true, proba >= 0.5),
        "best_threshold": best["threshold"],
        "profit_usd": best["profit"],
        "recall_at_threshold": recall_score(y_true, proba >= best["threshold"]),
        "customers_contacted": best["contacted"],
    }
    return {k: float(v) for k, v in metrics.items()}, curve


def predict_proba(model, df):
    """Probabilidad de churn con un modelo cargado desde MLflow."""
    proba = model.predict(clean(df)[FEATURES])
    return proba[:, 1] if getattr(proba, "ndim", 1) == 2 else proba
