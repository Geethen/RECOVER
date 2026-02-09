# RECOVER: Ecosystem Condition and Recovery Modeling

This project aims to extract ecological indices from Google Earth Engine (GEE) and train regression models to predict ecosystem conditions.

## Project Structure

```text
.
├── data/               # Dataset files (CSV, Shapefiles, GeoJSON)
├── notebooks/          # Jupyter notebooks for exploration and analysis
├── scripts/            # Python and JS scripts for data extraction and modeling
├── models/             # PyTorch model checkpoints
├── plots/              # Generated visualizations and performance plots
├── pyproject.toml      # Project metadata and dependencies
└── README.md           # Project overview and documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- Google Earth Engine account and project access

### Installation

1. Clone the repository.
2. Install dependencies using your preferred package manager (e.g., pip):
   ```bash
   pip install .
   ```

### Usage

#### 1. Data Extraction
Use `scripts/gee_extraction.py` to extract indices from GEE.
```bash
python scripts/gee_extraction.py
```

#### 2. Model Training
Train the multi-output regression model using `scripts/train_regression_model.py`.
```bash
python scripts/train_regression_model.py
```

#### 3. Evaluation
Evaluate the trained model using `scripts/evaluate_model.py`.
```bash
python scripts/evaluate_model.py
```

#### 4. Reference Condition Prediction
Predict reference states and uncertainty intervals using `scripts/predict_reference_conditions.py`.
```bash
python scripts/predict_reference_conditions.py
```

#### 5. Benchmarking
Select similar natural sites (benchmarks) and calculate the Benchmarked Condition Index (BCI) using `scripts/select_benchmark_sites.py`.
```bash
python scripts/select_benchmark_sites.py
```

#### 6. Visualization
Generate benchmarking distribution plots using `scripts/plot_benchmarking_results.py`.
```bash
python scripts/plot_benchmarking_results.py
```

## License
MIT
