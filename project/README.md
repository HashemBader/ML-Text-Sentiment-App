# setup and run instructions
# IMDb Text Sentiment - Course Project (Midpoint)

Repo layout:
- src/
  - data.py
  - features.py
  - train_baselines.py
  - evaluate.py
  - utils.py
- data/README.md (instructions to download the Kaggle IMDb dataset)
- models/ (trained model artifacts)
- mlruns/ (MLflow logs)

Quick run examples (from repo root):
1. Prepare data:
   python src/data.py --input_csv data/IMDB.csv --out_dir data/processed

2. Fit baselines:
   python src/train_baselines.py --data_dir data/processed --models_dir models

3. Evaluate:
   python src/evaluate.py --data_dir data/processed --models_dir models --output_dir reports

MLflow:
- MLflow tracking will log runs to local mlruns/ by default. Change MLFLOW_TRACKING_URI env var if needed.

See `data/README.md` for dataset download instructions.
