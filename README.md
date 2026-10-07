<div align="center">

# ✈️ Flight Price Prediction

### Explore an itinerary. Get a data-driven fare estimate.

[![Open the live app](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://flight-price-prediction-444.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-Project-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/Model-XGBoost-189AB4)](https://xgboost.readthedocs.io/)

[![Animated flight route: explore routes, estimate fares, powered by XGBoost](https://readme-typing-svg.demolab.com?font=Inter&weight=700&size=20&duration=2600&pause=900&color=38BDF8&center=true&vCenter=true&width=620&height=45&lines=Choose+a+route+%E2%9C%88%EF%B8%8F;Explore+your+fare+estimate+%F0%9F%93%8A;Powered+by+XGBoost+%F0%9F%A4%96)](https://flight-price-prediction-444.streamlit.app/)

**[Launch the live Flight Price Prediction Dashboard →](https://flight-price-prediction-444.streamlit.app/)**

</div>

---

> **Fare estimates, not live quotes.** This app predicts from historical training data. It does not fetch current airline prices, and estimates are not guaranteed fares. Actual prices vary with availability, booking time, and other market conditions.

## 🛫 What you can do

- Estimate a domestic flight fare from airline, origin, destination, class, stops, duration, date, and time-of-day.
- Inspect the 69-feature input sent to the trained XGBoost model.
- Follow an animated route visualization between the selected cities. The visualization loads external assets; prediction works independently.
- Try the [live Streamlit app](https://flight-price-prediction-444.streamlit.app/).

## 📊 Training dataset

The flight-price notebooks explore the `Clean_Dataset.csv` flight-fare data and engineer features for the prediction task. The final production-training cell uses the prepared **`Freshly_cleaned.csv`** dataset.

The cleaned training file contains **300,257 rows** before exact duplicate feature/price rows are removed; the published model was refit on **299,995 rows** and **69 encoded features**. The data includes airline, origin and destination, fare, class, journey day and month, flight number, duration, stop category, and departure/arrival time information. Flight number is used to group validation and test splits to reduce leakage, then excluded as an identifier from model features.

**The CSV files are not included in this repository.** You do not need the dataset to run the dashboard—the included model artifact is sufficient. To retrain, obtain the dataset from a source you are authorized to use and place `Freshly_cleaned.csv` in the project root or a `data/` folder. The notebook also contains exploratory cells that may need their input paths updated. Do not publish datasets unless you have permission to redistribute them.

## 📈 Model evaluation

The training notebook compares Random Forest and XGBoost using validation mean absolute error (MAE), evaluates the selected model on untouched flight-number groups, and then refits it on the available cleaned data. XGBoost was selected for the current artifact.

| Held-out test metric | Result |
| --- | ---: |
| Mean absolute error (MAE) | INR 2,553.91 |
| R² | 0.9509 |

These results describe one evaluation split from this dataset; they do not guarantee accuracy for future or live fares. A realistic prediction is not proof of correctness—compare against a real quote for the same itinerary to evaluate an individual estimate.

## 🗺️ Project map

```text
.
├── app/
│   └── app.py                                  # Streamlit interface and prediction flow
├── EDA/
│   ├── EDA And Feature Engineering Flight Price Dataset.ipynb
│   └── EDA.IPYNB                              # Separate red-wine exploration
├── final_flight_price_rf_model.pkl             # Included XGBoost artifact
├── requirements.txt
└── README.md
```

The model filename retains “rf” from an earlier training workflow; the included artifact is **XGBoost**, not Random Forest. The former 1.05 GB `flight_price_rf_model.pkl` artifact is intentionally not included in the repository because it exceeds GitHub's regular-file size limit.

## ⚡ Run it locally

### 1. Install dependencies

From the project root in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If environment activation is blocked, run pip using the virtual environment directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Start the dashboard

```powershell
streamlit run app/app.py
```

The app uses `final_flight_price_rf_model.pkl` by default. To select a different trusted artifact, set `MODEL_PATH` first:

```powershell
$env:MODEL_PATH = "C:\path\to\model.pkl"
streamlit run app/app.py
```

The app checks the configured model path and supported repository-relative fallback locations. Run Streamlit from the project root so relative paths resolve as expected.

## 🧪 Train or explore

Open the desired notebook in Jupyter or VS Code using the environment where the requirements are installed. For production retraining, supply `Freshly_cleaned.csv` in the project root or `data/` directory; update the notebook's candidate paths if the file is elsewhere. Some earlier exploratory cells may still refer to local paths or `Clean_Dataset.csv`.

`EDA/EDA.IPYNB` is a separate red-wine EDA notebook and fetches its dataset from UCI, so that notebook requires internet access.

## 🚀 Deployment notes

- The model artifact is loaded with joblib. Only use pickle/joblib files from trusted sources; loading an untrusted artifact can execute code.
- Keep large datasets and model files out of Git unless repository policy explicitly permits them. Use an approved artifact store for files that exceed GitHub's size limit.
- Use compatible Python and XGBoost versions when loading the model. `requirements.txt` provides dependency constraints, not a fully locked environment.
- The route animation uses external browser assets and needs internet access; fare prediction does not depend on those assets.
- If you replace the model, verify that its feature count and schema match the app before deployment.

## 🔧 Troubleshooting

| Issue | What to check |
| --- | --- |
| Model is disconnected | Confirm the model file is present, trusted, and `MODEL_PATH` points to it if using a non-default location. |
| Model load or prediction error | Check Python/XGBoost compatibility and ensure the artifact matches the app's 69-feature schema. |
| Training notebook cannot find data | Place `Freshly_cleaned.csv` in the project root or `data/`, or update the notebook's candidate paths. |
| Wine notebook cannot fetch data | Check internet access to the UCI dataset. |
| Route animation is blank | Check browser access to external visualization assets; the fare prediction is separate. |

---

<div align="center">

**Ready for take-off?** [Open the live fare estimator ✈️](https://flight-price-prediction-444.streamlit.app/)

</div>
