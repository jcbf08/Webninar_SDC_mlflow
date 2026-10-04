"""PASO 4 · PRODUCCIÓN: el quality gate decide si @challenger pasa a @champion.

    python src/promote.py
"""

import sys

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from common import CFG, evaluate, load_data, predict_proba, setup_mlflow


def score(alias, X_test, y_test):
    model = mlflow.pyfunc.load_model(f"models:/{CFG['model_name']}@{alias}")
    return evaluate(y_test, predict_proba(model, X_test))[0]


def main():
    setup_mlflow()
    name, gate, client = CFG["model_name"], CFG["gate"], MlflowClient()
    _, X_test, _, y_test = load_data()

    try:
        challenger = client.get_model_version_by_alias(name, "challenger")
    except MlflowException:
        print("No hay @challenger. Ejecuta primero: make register")
        return 1
    ch = score("challenger", X_test, y_test)
    print(f"\nChallenger v{challenger.version}: AUC {ch['test_auc']:.4f} · "
          f"utilidad USD {ch['profit_usd']:,.0f}")

    checks = {
        f"AUC >= {gate['min_auc']}": ch["test_auc"] >= gate["min_auc"],
        f"recall >= {gate['min_recall']}": ch["recall_at_threshold"] >= gate["min_recall"],
    }
    try:
        champion = client.get_model_version_by_alias(name, "champion")
        cp = score("champion", X_test, y_test)
        print(f"Champion   v{champion.version}: AUC {cp['test_auc']:.4f} · "
              f"utilidad USD {cp['profit_usd']:,.0f}")
        if champion.version != challenger.version:
            checks[f"mejora vs champion >= {gate['min_gain']}"] = (
                ch["test_auc"] - cp["test_auc"] >= gate["min_gain"])
    except MlflowException:
        print("Champion   : (ninguno todavía)")

    print("\nQuality gate")
    for label, ok in checks.items():
        print(f"  [{'PASA' if ok else 'FALLA'}] {label}")
    approved = all(checks.values())

    # La decisión queda auditada como un run más del experimento
    with mlflow.start_run(run_name=f"gate-v{challenger.version}"):
        mlflow.set_tags({"stage": "gate", "decision": "approved" if approved else "rejected"})
        mlflow.log_metrics({f"challenger_{k}": v for k, v in ch.items()})

    client.set_model_version_tag(name, challenger.version, "status",
                                 "approved" if approved else "rejected")
    if not approved:
        print("\nRECHAZADO: el champion actual se mantiene en producción.")
        return 2
    client.set_registered_model_alias(name, "champion", challenger.version)
    print(f"\nPROMOVIDO: models:/{name}@champion -> versión {challenger.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
