# Walmart Sales Intelligence Dashboard

An interactive Streamlit dashboard for weekly sales across 45 Walmart stores, with executive briefs, rich charts, a Random Forest sales forecast with explanations, and a downloadable HTML report.

## Features

- **Executive Brief:** auto-generated insight cards and recommended actions that update with the filters.
- **Dashboard:** revenue trend with holiday markers and range selector, store-type share, top departments, store ranking, holiday vs regular distributions, monthly seasonality, heatmaps, correlation matrix, and distributions.
- **Predictive Analysis:** a Random Forest model validated on the most recent 15% of weeks. Pick a store, department, month, and holiday flag to get a live forecast with a likely range, a gauge against historical averages, and a "why this prediction" breakdown.
- **Data Explorer:** browse filtered records and download an aggregated CSV.
- **Summary Report:** narrative summary and a downloadable HTML report with embedded charts.

## Quick start

Requires Python 3.9 or newer.

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/walmart-sales-dashboard-advanced.git
cd walmart-sales-dashboard-advanced

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add the data file
#    Copy processed_walmart_sales.csv into the data/ folder

# 5. Run the app
streamlit run app.py
```

The app opens at http://localhost:8501. If the CSV is missing, you can upload it from the page.

## Notes

- The first load trains the model, which takes about 20 to 30 seconds. Results are cached afterwards.
- The forecast's explanation is approximate. It shows how each input moves the prediction relative to a typical value.
- Use forecasts for planning ranges rather than exact figures.

## Project structure

```
walmart-sales-dashboard-advanced/
├── app.py              # Streamlit app: data, model, charts, and report
├── requirements.txt    # Python dependencies
├── data/               # Place the CSV here (not committed)
├── docs/               # Screenshots and documentation assets
├── .gitignore
└── README.md
```

## License

Add a license here if you plan to share the project publicly.
