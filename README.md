# Flight Price Prediction Dashboard

A Streamlit dashboard that estimates an Indian domestic flight fare from itinerary details using a pre-trained XGBoost model. The repository also contains notebooks for flight-price feature engineering and an independent red-wine exploratory data analysis.

[![Open the live app](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://flight-price-prediction-444.streamlit.app/)

**Live app:** [Flight Price Prediction Dashboard](https://flight-price-prediction-444.streamlit.app/)

> **Prediction disclaimer:** Estimates are model outputs for exploration and should not be treated as live quotes or guaranteed fares. Prices change with availability, booking time, and other factors that are not represented in the model.

## Features

- Predicts an estimated fare from airline, route, ticket class, stops, duration, date, and time-of-day inputs.
- Builds the verified 69-feature model input schema used to train the included model. When a model contains feature-name metadata, the app uses it directly.
- Shows the generated feature vector alongside the prediction.
- Includes an animated route visualization. Its external map and animation assets require an internet connection; fare prediction itself does not.

## Repository layout

```text
.
├── app/
│   └── app.py
├── EDA/
│   ├── EDA And Feature Engineering Flight Price Dataset.ipynb
│   └── EDA.IPYNB
├── final_flight_price_rf_model.pkl
├── flight_price_rf_model.pkl
├── requirements.txt
└── README.md
```

- `app/app.py` is the Streamlit application.
- `final_flight_price_rf_model.pkl` is the XGBoost model artifact included in the repository (the filename is retained from the training notebook). The 1.05 GB `flight_price_rf_model.pkl` artifact is intentionally excluded from GitHub because GitHub rejects regular Git files over 100 MB. The app checks `MODEL_PATH` first, then `flight_price_rf_model.pkl`, `final_flight_price_rf_model.pkl`, and a repository-relative EDA path. Supply the larger artifact separately if you specifically need it.
- `EDA/EDA And Feature Engineering Flight Price Dataset.ipynb` explores and engineers features from `Clean_Dataset.csv`. The CSV is not included and must be supplied separately.
- `EDA/EDA.IPYNB` is a separate red-wine EDA notebook. It reads the red-wine dataset directly from the UCI URL, so running that cell requires internet access.

## Requirements

- Python with a version supported by the installed NumPy, pandas, scikit-learn, Streamlit, and notebook packages.
- `requirements.txt` installs the packages used by the dashboard and notebooks.
- A compatible model pickle is required to make predictions. The included `final_flight_price_rf_model.pkl` is the default available artifact in a fresh checkout. Serialized XGBoost models are not guaranteed to work across library versions; use the same XGBoost and Python versions used to train the artifact.

The dependency constraints in `requirements.txt` are minimum versions, not a lockfile. For repeatable deployments, resolve and lock the exact package versions tested with the model artifacts.

## Model validation

The included model was retrained using `Freshly_cleaned.csv`. The notebook removes the identifier columns, drops exact duplicate feature/price rows, and uses a flight-number-grouped validation and test split to reduce leakage between the same flights. It compares Random Forest and XGBoost using fare MAE, then evaluates the selected model once on the untouched test groups before refitting it on all available rows.

For the current local dataset, XGBoost was selected. On the untouched grouped test set it achieved **INR 2,553.91 MAE** and **R² 0.9509**. These are dataset-specific evaluation results, not a guarantee for future fares. The notebook checks that all 69 encoded features and their order match the Streamlit application before it saves the model.

## Local setup

Run these commands from the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, either enable the appropriate local script policy for your environment or invoke the virtual-environment Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the dashboard

From the repository root:

```powershell
streamlit run app/app.py
```

To select a model explicitly, set `MODEL_PATH` to the absolute path of a trusted model artifact before starting Streamlit:

```powershell
$env:MODEL_PATH = "F:\plane\final_flight_price_rf_model.pkl"
streamlit run app/app.py
```

The application also checks the current working directory for `flight_price_rf_model.pkl` and `final_flight_price_rf_model.pkl`. Start it from the repository root so those fallback paths resolve as expected.

## Run the notebooks

Install the requirements, open the desired notebook in Jupyter or VS Code, and select the Python environment where the requirements were installed. The final production-training cell requires the cleaned `Freshly_cleaned.csv` dataset and searches project `data/` folders and the original local Downloads path; update its candidate paths if your CSV is stored elsewhere. Earlier exploratory cells may also contain machine-specific paths. Do not commit private or licensed datasets unless you have permission to distribute them.

## Model and data handling

- Only load pickle/joblib files from trusted sources. Deserializing an untrusted model file can execute arbitrary code.
- Keep large datasets and model artifacts out of source control unless repository policy explicitly allows them. The oversized `flight_price_rf_model.pkl` is ignored by Git; distribute it through an approved artifact store if needed.
- Record the model training code, source data provenance, and exact dependency versions when publishing or deploying a new model.
- Verify that a replacement artifact has the feature metadata expected by the app before deployment.

## Troubleshooting

- **Model disconnected:** confirm `MODEL_PATH` points to an existing, trusted `.pkl` artifact, or use the included `final_flight_price_rf_model.pkl` from the repository root.
- **Model load or prediction error:** check Python and XGBoost compatibility against the model's training environment.
- **Flight notebook cannot find data:** place `Clean_Dataset.csv` in a searched location and update any later machine-specific paths in the notebook.
- **Wine notebook cannot fetch data:** check internet access to the UCI dataset URL.
- **Route animation is blank:** check browser network access to the external visualization assets; the prediction feature is separate.
