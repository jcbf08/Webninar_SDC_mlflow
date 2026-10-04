# MLflow de Cero a Producción

Laboratorio del webinar: **trackea, compara y registra tus experimentos como un pro.**

Caso: predecir qué clientes de una telco se van (churn) y decidir a quién contactar para que
la campaña de retención gane dinero. Todo el ciclo queda gobernado por MLflow.

## Estructura

```
├── config.yaml        Parámetros, supuestos de negocio y reglas del quality gate
├── data/churn.csv     7,043 clientes (dataset público Telco Customer Churn de IBM)
├── src/
│   ├── common.py      Datos, pipeline y métricas (compartido)
│   ├── train.py       1 · Trackea
│   ├── tune.py        2 · Compara
│   ├── register.py    3 · Registra
│   ├── promote.py     4 · Quality gate
│   └── api.py         5 · API con el modelo en producción
├── tests/test_lab.py  Tests de datos, métrica de negocio y flujo MLflow completo
├── Makefile           Un comando por paso
└── GUIA.md            Guion de la demo en 20 minutos
```

## Cómo ejecutarlo en GitHub Codespaces

1. En GitHub: **Code ▸ Codespaces ▸ Create codespace on main** (instala todo solo).
2. Terminal 1: `make ui` y abre el puerto 5000 desde la pestaña **Ports**.
3. Terminal 2:

| Comando | Qué hace | Tiempo |
|---|---|---|
| `make train` | Entrena 3 modelos y registra todo en MLflow | 30 s |
| `make tune` | 25 pruebas de Optuna como runs hijos | 70 s |
| `make register` | El mejor run pasa al Model Registry como `@challenger` | 3 s |
| `make promote` | El quality gate lo promueve a `@champion` | 5 s |
| `make reject` | Un modelo malo intenta llegar a producción y el gate lo frena | 15 s |
| `make serve` | API en el puerto 8000 (abre `/docs` automáticamente) | – |

`make all` ejecuta los pasos 1 a 4 de una vez (menos de 2 minutos). `make clean` deja todo en cero.

## Resultados de referencia

Holdout de 1,409 clientes, semilla 42:

| Modelo | AUC (CV) | AUC (test) | Utilidad de campaña |
|---|---|---|---|
| Contactar a todos (sin modelo) | – | – | −19,090 USD |
| logreg | 0.8460 | 0.8416 | 17,835 USD |
| forest | 0.8480 | 0.8420 | 18,405 USD |
| boosting (default) | 0.8348 | 0.8295 | 17,250 USD |
| **boosting + Optuna** | **0.8492** | **0.8442** | **18,925 USD** |

La utilidad usa los supuestos de `config.yaml` (retener un cliente vale 500 USD, la oferta
cuesta 60 USD, 35 % de aceptación). Son ilustrativos: cámbialos por los de tu negocio.
