# 📈 RAM Price Predictor (End-to-End ML)

This project predicts the market price of RAM (Random Access Memory) based on technical specifications like capacity, frequency, latency, and generation (DDR4 vs DDR5). 

## 🚀 Key Features
- **End-to-End Workflow:** From framing the problem to model evaluation.
- **Stratified Sampling:** Ensures the test set is representative of the most important feature (e.g., DDR Generation).
- **Scikit-Learn Pipelines:** Automated data preprocessing using `ColumnTransformer` for a clean, reproducible flow.
- **Baseline Modeling:** Implementation of **Random Forest Regression** to establish performance benchmarks.

## 🛠️ Tech Stack
- **Languages:** Python 3.10+
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn

## 📊 The Workflow
1. **Frame the Problem:** Regression task with RMSE (Root Mean Square Error) as the primary metric.
2. **Get the Data:** Simulated/Scraped RAM pricing datasets.
3. **Explore the Data (EDA):** Visualizing correlations between frequency/latency and price.
4. **Prepare the Data:** - Handling missing values with `SimpleImputer`.
    - Encoding categorical specs with `OneHotEncoder`.
    - Feature scaling via `StandardScaler`.
5. **Select and Train:** Training a Random Forest model.
6. **Fine-Tune:** Evaluating performance using Cross-Validation.

## ⚙️ Installation & Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/gabegaines1024/RAM-Predictions]
