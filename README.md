# Flight Price Prediction Dashboard

A Streamlit dashboard that estimates an Indian domestic flight fare from itinerary details using a pre-trained Random Forest model. The repository also contains notebooks for flight-price feature engineering and an independent red-wine exploratory data analysis.

> **Prediction disclaimer:** Estimates are model outputs for exploration and should not be treated as live quotes or guaranteed fares. Prices change with availability, booking time, and other factors that are not represented in the model.

## Features

- Predicts an estimated fare from airline, route, ticket class, stops, duration, date, and time-of-day inputs.
- Builds model features from the model's `feature_names_in_` metadata.
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
- `final_flight_price_rf_model.pkl` is the model artifact included in the repository. The 1.05 GB `flight_price_rf_model.pkl` artifact is intentionally excluded from GitHub because GitHub rejects regular Git files over 100 MB. The app checks `MODEL_PATH` first, then `flight_price_rf_model.pkl`, `final_flight_price_rf_model.pkl`, and a repository-relative EDA path. Supply the larger artifact separately if you specifically need it.
- `EDA/EDA And Feature Engineering Flight Price Dataset.ipynb` explores and engineers features from `Clean_Dataset.csv`. The CSV is not included and must be supplied separately.
- `EDA/EDA.IPYNB` is a separate red-wine EDA notebook. It reads the red-wine dataset directly from the UCI URL, so running that cell requires internet access.

## Requirements

- Python with a version supported by the installed NumPy, pandas, scikit-learn, Streamlit, and notebook packages.
- `requirements.txt` installs the packages used by the dashboard and notebooks.
- A compatible model pickle is required to make predictions. The included `final_flight_price_rf_model.pkl` is the default available artifact in a fresh checkout. Serialized scikit-learn models are not guaranteed to work across library versions; use the same scikit-learn (and, where applicable, NumPy) versions used to train each artifact.

The dependency constraints in `requirements.txt` are minimum versions, not a lockfile. For repeatable deployments, resolve and lock the exact package versions tested with the model artifacts.

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

Install the requirements, open the desired notebook in Jupyter or VS Code, and select the Python environment where the requirements were installed. The flight-price notebook initially searches for `Clean_Dataset.csv` in several local locations, but later cells also contain machine-specific absolute paths; update those cells to your dataset location before running the notebook from top to bottom. Do not commit private or licensed datasets unless you have permission to distribute them.

## Model and data handling

- Only load pickle/joblib files from trusted sources. Deserializing an untrusted model file can execute arbitrary code.
- Keep large datasets and model artifacts out of source control unless repository policy explicitly allows them. The oversized `flight_price_rf_model.pkl` is ignored by Git; distribute it through an approved artifact store if needed.
- Record the model training code, source data provenance, and exact dependency versions when publishing or deploying a new model.
- Verify that a replacement artifact has the feature metadata expected by the app before deployment.

## Troubleshooting

- **Model disconnected:** confirm `MODEL_PATH` points to an existing, trusted `.pkl` artifact, or use the included `final_flight_price_rf_model.pkl` from the repository root.
- **Model load or prediction error:** check Python and scikit-learn compatibility against the model's training environment.
- **Flight notebook cannot find data:** place `Clean_Dataset.csv` in a searched location and update any later machine-specific paths in the notebook.
- **Wine notebook cannot fetch data:** check internet access to the UCI dataset URL.
- **Route animation is blank:** check browser network access to the external visualization assets; the prediction feature is separate.
