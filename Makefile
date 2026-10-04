.DEFAULT_GOAL := help
export MLFLOW_DISABLE_AGENT_HINT := 1

help:  ## Muestra esta ayuda
	@grep -E '^[a-z-]+:.*## ' Makefile | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup:  ## Instala dependencias
	python -m pip install -q -r requirements.txt

ui:  ## Interfaz de MLflow en el puerto 5000
	mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000 --allowed-hosts "*" --cors-allowed-origins "*"

train:  ## 1 Trackea: entrena 3 modelos base
	python src/train.py

tune:  ## 2 Compara: 25 pruebas de Optuna como runs hijos
	python src/tune.py

register:  ## 3 Registra: mejor run -> @challenger
	python src/register.py

promote:  ## 4 Quality gate: @challenger -> @champion
	python src/promote.py

reject:  ## Demo: un modelo malo intenta llegar a producción
	python src/train.py --model forest -p n_estimators=10 -p max_depth=1
	python src/register.py --latest
	-python src/promote.py

serve:  ## 5 API con el @champion en el puerto 8000
	@echo "Abre el puerto 8000 desde la pestaña Ports (la raíz te lleva a /docs)"
	python -m uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000

all: train tune register promote  ## Pasos 1 a 4 de una vez

test:  ## Tests y linter
	python -m ruff check .
	python -m pytest -q

clean:  ## Empezar de cero
	rm -rf mlflow.db mlruns mlartifacts .pytest_cache

.PHONY: help setup ui train tune register promote reject serve all test clean
