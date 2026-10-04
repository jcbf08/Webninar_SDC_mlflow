"""PASO 2 · COMPARA: Optuna busca hiperparámetros; cada prueba es un run hijo.

    python src/tune.py
"""

import mlflow
import optuna
from sklearn.model_selection import cross_val_score

from common import CFG, CV, build_pipeline, load_data, setup_mlflow
from train import train_and_log


def main():
    setup_mlflow()
    X_train, X_test, y_train, y_test = load_data()

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 64),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
        }
        # Run HIJO: aparece anidado bajo el run padre en la interfaz
        with mlflow.start_run(run_name=f"trial-{trial.number:02d}", nested=True):
            mlflow.set_tag("stage", "trial")
            mlflow.log_params(params)
            score = cross_val_score(build_pipeline("boosting", params), X_train, y_train,
                                    cv=CV, scoring="roc_auc", n_jobs=-1).mean()
            mlflow.log_metric("cv_auc", score)
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with mlflow.start_run(run_name="optuna-tuning"):
        mlflow.set_tag("stage", "tuning")
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=CFG["seed"]))
        study.optimize(objective, n_trials=CFG["n_trials"])
        mlflow.log_metric("best_cv_auc", study.best_value)

        # Reentrena el ganador y lo registra completo (modelo + gráficos)
        _, m = train_and_log("boosting", study.best_params, X_train, X_test, y_train, y_test,
                             run_name="boosting-tuned", nested=True, stage="tuned")

    print(f"Mejor CV AUC : {study.best_value:.4f}")
    print(f"Test AUC     : {m['test_auc']:.4f}")
    print(f"Utilidad     : USD {m['profit_usd']:,.0f} (umbral {m['best_threshold']:.2f})")


if __name__ == "__main__":
    main()
