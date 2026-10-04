"""PASO 5 · CONSUME: API que sirve el modelo @champion.

    make serve   ->   http://localhost:8000/docs
"""

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from pydantic import BaseModel

from common import CFG, predict_proba, setup_mlflow

app = FastAPI(title="Churn API")
state = {}


def load_champion():
    """Carga por ALIAS: este código nunca conoce el número de versión."""
    setup_mlflow()
    name = CFG["model_name"]
    version = MlflowClient().get_model_version_by_alias(name, "champion")
    state["model"] = mlflow.pyfunc.load_model(f"models:/{name}@champion")
    state["version"] = version.version
    state["threshold"] = float(version.tags.get("threshold", 0.5))


class Customer(BaseModel):
    gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    tenure: float = 2
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "Yes"
    StreamingMovies: str = "Yes"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 95.5
    TotalCharges: float = 191.0


@app.get("/", include_in_schema=False)
def root():
    """La raíz lleva directo a la página interactiva."""
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    if "model" not in state:
        try:
            load_champion()
        except MlflowException as exc:
            return {"status": "sin modelo", "detail": str(exc)}
    return {"status": "ok", "model_version": state["version"], "threshold": state["threshold"]}


@app.post("/reload")
def reload():
    """Después de promover un modelo nuevo: recarga sin redeploy."""
    load_champion()
    return health()


@app.post("/predict")
def predict(customer: Customer):
    if health()["status"] != "ok":
        raise HTTPException(503, "No hay modelo @champion. Ejecuta: make promote")
    proba = float(predict_proba(state["model"], pd.DataFrame([customer.model_dump()]))[0])
    return {"churn_probability": round(proba, 4),
            "contact_customer": proba >= state["threshold"],
            "threshold": state["threshold"],
            "model_version": state["version"]}
