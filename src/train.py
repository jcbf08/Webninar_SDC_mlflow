"""PASO 1 · TRACKEA: entrena modelos y registra todo en MLflow.

    python src/train.py                                  # los 3 modelos base
    python src/train.py --model forest -p max_depth=1 -p n_estimators=10
"""

import argparse
import ast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score

from common import CV, MODELS, build_pipeline, evaluate, load_data, setup_mlflow


def train_and_log(name, params, X_train, X_test, y_train, y_test, run_name=None,
                  nested=False, stage="baseline"):
    pipeline = build_pipeline(name, params)

    with mlflow.start_run(run_name=run_name or name, nested=nested) as run:
        # 1) TAGS: contexto para filtrar y auditar
        mlflow.set_tags({"model_family": name, "stage": stage, "candidate": "true"})

        # 2) PARAMS: hiperparámetros
        mlflow.log_params({**MODELS[name][1], **(params or {})})

        # 3) DATASET: con qué datos exactos se entrenó (guarda una huella)
        train_df = pd.concat([X_train, y_train.rename("Churn")], axis=1)
        mlflow.log_input(mlflow.data.from_pandas(train_df, name="churn-train",
                                                 targets="Churn"), context="training")

        # 4) MÉTRICAS: validación cruzada + test + negocio
        cv = cross_val_score(pipeline, X_train, y_train, cv=CV,
                             scoring="roc_auc", n_jobs=-1)
        mlflow.log_metric("cv_auc", cv.mean())
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        metrics, curve = evaluate(y_test, proba)
        mlflow.log_metrics(metrics)

        # 5) ARTEFACTOS: gráficos como evidencia
        for path, fig in make_plots(y_test, proba, curve, name).items():
            mlflow.log_figure(fig, path)
            plt.close(fig)

        # 6) MODELO: pipeline completo + firma + ejemplo de entrada
        info = mlflow.sklearn.log_model(
            pipeline, name="model",
            signature=infer_signature(X_test, proba),
            input_example=X_test.head(3),
            pyfunc_predict_fn="predict_proba",
            serialization_format="cloudpickle",
        )
        mlflow.set_tag("model_uri", info.model_uri)

    metrics["cv_auc"] = float(cv.mean())
    return run.info.run_id, metrics


def make_plots(y_test, proba, curve, title):
    fig1, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, proba >= 0.5, cmap="Blues",
                                            display_labels=["Se queda", "Churn"],
                                            colorbar=False, ax=ax)
    ax.set_title(f"Matriz de confusión · {title}")

    best = curve.loc[curve["profit"].idxmax()]
    fig2, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve["threshold"], curve["profit"], color="#0A7EA4", lw=2)
    ax.axvline(best["threshold"], color="#F59E0B", ls="--")
    ax.set_title(f"Utilidad máxima: USD {best['profit']:,.0f} con umbral {best['threshold']:.2f}")
    ax.set_xlabel("Umbral de decisión")
    ax.set_ylabel("Utilidad esperada (USD)")
    ax.grid(alpha=0.3)

    for fig in (fig1, fig2):
        fig.tight_layout()
    return {"confusion_matrix.png": fig1, "profit_curve.png": fig2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[*MODELS, "all"], default="all")
    parser.add_argument("-p", "--param", action="append", default=[], help="clave=valor")
    args = parser.parse_args()
    params = {k: ast.literal_eval(v) for k, v in (p.split("=", 1) for p in args.param)}

    setup_mlflow()
    data = load_data()
    print(f"\n{'modelo':<12}{'cv_auc':>8}{'test_auc':>10}{'recall':>8}{'utilidad USD':>14}")
    for name in (MODELS if args.model == "all" else [args.model]):
        _, m = train_and_log(name, params or None, *data)
        print(f"{name:<12}{m['cv_auc']:>8.4f}{m['test_auc']:>10.4f}"
              f"{m['test_recall']:>8.3f}{m['profit_usd']:>14,.0f}")


if __name__ == "__main__":
    main()
