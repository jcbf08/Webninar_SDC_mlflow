# Guion de la demo · 20 minutos

## Antes del webinar

- Crea el Codespace con tiempo (la primera instalación tarda 2 a 3 minutos).
- Ejecuta `make clean` para empezar con MLflow vacío.
- Terminal 1: `make ui`. Abre el puerto 5000 (pestaña **Ports ▸ Open in Browser**).
- Plan B: un segundo Codespace con `make all` ya ejecutado.

## Paso a paso

| Min | Paso | Comando | Qué mostrar |
|---|---|---|---|
| 0–2 | Contexto | – | `config.yaml` y la carpeta `src/`: un archivo por paso |
| 2–6 | 1 · Trackea | `make train` | `src/train.py`: los 6 bloques numerados. En MLflow: un run con sus parámetros, métricas, dataset y la pestaña **Artifacts** (`profit_curve.png`) |
| 6–11 | 2 · Compara | `make tune` | Mientras corre, explica Optuna. En MLflow: expande `optuna-tuning` con **+**, selecciona runs ▸ **Compare**, busca `metrics.test_auc > 0.84` |
| 11–13 | 3 · Registra | `make register` | **Models ▸ telco-churn-classifier**: versión 1 con alias `@challenger` y enlace al run |
| 13–16 | 4 · Gate | `make promote` y luego `make reject` | Terminal: PASA / FALLA. En MLflow: v1 es `@champion`, v2 queda rechazada |
| 16–19 | 5 · Consume | `make serve` | Puerto 8000 (abre `/docs` solo) ▸ `GET /health` ▸ `POST /predict` con el ejemplo: prob. ≈ 0.90, contactar. Cambia `Contract` a `Two year` y `tenure` a 60: baja a ≈ 0.20 |
| 19–20 | Cierre | – | `.github/workflows/ci.yml`: el mismo gate corre en cada push |

## Ideas que deben quedar

1. Con pocas líneas de MLflow, cada entrenamiento deja evidencia completa.
2. Comparar es una consulta, no un Excel.
3. La métrica de negocio (utilidad y umbral) se registra junto a las técnicas.
4. Producción carga `models:/telco-churn-classifier@champion`: nunca un número de versión.
5. Promover es una regla escrita en `config.yaml`, no una decisión manual.

## Si algo falla

| Síntoma | Solución |
|---|---|
| La interfaz no carga | Ábrela desde la pestaña **Ports**, no escribiendo la URL |
| `No hay @challenger` | Ejecuta `make register` antes de `make promote` |
| La API dice `sin modelo` | Ejecuta `make promote` y luego `POST /reload` |
| Quiero empezar de cero | `make clean` |
