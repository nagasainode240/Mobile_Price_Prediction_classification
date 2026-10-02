# Mobile Price Prediction - MLOps Project (CLA 2)

Predicts a phone's price range (0-3) from its specifications (Kaggle "Mobile Price Classification", 2000 rows, 20 features).
Extends the CLA 1 notebook (`notebooks/project_implementation.ipynb`) into a runnable MLOps project.

## Setup
```
pip install -r requirements.txt
```
Place `train.csv` in `data/raw/` (already included).

## Run (from the project folder)
```
python .\pipelines\run_lab3_baseline.py   # Lab 3: preprocess -> train -> evaluate
python .\pipelines\run_lab4_tracking.py   # Lab 4: same run + experiment tracking CSV and logs
python .\pipelines\run_lab5_pipeline.py   # Lab 5: schema validation -> sklearn pipeline -> output + reproducibility checks
python .\pipelines\run_lab6_registry.py   # Lab 6: MLflow registry -> promote champion -> deployment report
```
View the MLflow registry: `mlflow ui --backend-store-uri sqlite:///mlflow.db`

## Structure
| Folder | Purpose |
|---|---|
| `data/raw`, `data/processed` | input data, processed arrays, dataset_metadata.json |
| `src/` | preprocess / train / evaluate / predict (Lab 3-4) and validation + registry scripts (Lab 5-6) |
| `pipelines/` | one runner per lab |
| `models/` | model.pkl, preprocessor.pkl |
| `outputs/`, `reports/` | confusion matrix, evaluation report |
| `logs/`, `artifacts/` | run logs, tracking CSV, validation/reproducibility/registry JSON reports |
