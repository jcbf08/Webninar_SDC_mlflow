"""PASO 3 · REGISTRA: lleva un run al Model Registry como @challenger.

    python src/register.py            # el MEJOR run por test_auc
    python src/register.py --latest   # el run MÁS RECIENTE (demo del quality gate)
"""

import argparse

import mlflow
from mlflow import MlflowClient

from common import CFG, setup_mlflow


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--latest", action="store_true")
    args = parser.parse_args(argv)

    setup_mlflow()
    name = CFG["model_name"]

    # La misma consulta que escribirías en el buscador de la interfaz
    order = "attributes.start_time DESC" if args.latest else "metrics.test_auc DESC"
    runs = mlflow.search_runs(experiment_names=[CFG["experiment"]],
                              filter_string="tags.candidate = 'true'",
                              order_by=[order], max_results=1)
    if runs.empty:
        raise SystemExit("No hay runs. Ejecuta primero: make train")
    best = runs.iloc[0]
    print(f"Run elegido: {best['tags.mlflow.runName']} · test_auc={best['metrics.test_auc']:.4f}")

    version = mlflow.register_model(best["tags.model_uri"], name).version
    client = MlflowClient()
    client.set_model_version_tag(name, version, "threshold", str(best["metrics.best_threshold"]))
    client.set_model_version_tag(name, version, "status", "pending")
    client.set_registered_model_alias(name, "challenger", version)
    print(f"Registrado: models:/{name}@challenger -> versión {version}")


if __name__ == "__main__":
    main()
