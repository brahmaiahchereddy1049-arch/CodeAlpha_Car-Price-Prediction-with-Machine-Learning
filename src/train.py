"""
Train and evaluate car price prediction models.

Run from the project root:
    python src/train.py
"""

from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "car_data.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"

MODEL_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)

    # Remove duplicate records.
    df = df.drop_duplicates().copy()

    # Basic validation.
    required = {
        "Year", "Selling_Price", "Present_Price", "Driven_kms",
        "Fuel_Type", "Selling_type", "Transmission", "Owner"
    }
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    # Feature engineering.
    current_year = pd.Timestamp.today().year
    df["Car_Age"] = current_year - df["Year"]

    # Guard against impossible negative ages.
    df.loc[df["Car_Age"] < 0, "Car_Age"] = 0

    return df


def build_preprocessor():
    categorical_features = ["Fuel_Type", "Selling_type", "Transmission"]
    numeric_features = ["Present_Price", "Driven_kms", "Car_Age", "Owner"]

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            ("numeric", "passthrough", numeric_features),
        ]
    )


def main():
    df = load_and_prepare_data()

    feature_columns = [
        "Present_Price",
        "Driven_kms",
        "Car_Age",
        "Fuel_Type",
        "Selling_type",
        "Transmission",
        "Owner",
    ]

    X = df[feature_columns]
    y = df["Selling_Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            random_state=42, max_depth=8
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            max_depth=12,
            n_jobs=-1,
        ),
    }

    results = []
    trained = {}

    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        results.append(
            {
                "Model": name,
                "MAE": mae,
                "RMSE": rmse,
                "R2 Score": r2,
            }
        )

        trained[name] = pipeline

    results_df = (
        pd.DataFrame(results)
        .sort_values("R2 Score", ascending=False)
        .reset_index(drop=True)
    )

    results_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    best_name = results_df.loc[0, "Model"]
    best_model = trained[best_name]
    best_predictions = best_model.predict(X_test)

    joblib.dump(
        best_model,
        MODEL_DIR / "car_price_model.joblib",
    )

    # Plot 1: Actual vs predicted.
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, best_predictions, alpha=0.7)
    minimum = min(y_test.min(), best_predictions.min())
    maximum = max(y_test.max(), best_predictions.max())
    plt.plot([minimum, maximum], [minimum, maximum])
    plt.xlabel("Actual Selling Price")
    plt.ylabel("Predicted Selling Price")
    plt.title(f"Actual vs Predicted Prices - {best_name}")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

    # Plot 2: Selling-price distribution.
    plt.figure(figsize=(8, 6))
    plt.hist(df["Selling_Price"], bins=25)
    plt.xlabel("Selling Price")
    plt.ylabel("Number of Cars")
    plt.title("Distribution of Car Selling Prices")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "price_distribution.png", dpi=150)
    plt.close()

    # Plot 3: Present price vs selling price.
    plt.figure(figsize=(8, 6))
    plt.scatter(df["Present_Price"], df["Selling_Price"], alpha=0.7)
    plt.xlabel("Present Price")
    plt.ylabel("Selling Price")
    plt.title("Present Price vs Selling Price")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "present_vs_selling_price.png", dpi=150)
    plt.close()

    print("\nModel Evaluation")
    print("=" * 60)
    print(results_df.round(4).to_string(index=False))
    print("\nBest model:", best_name)
    print("Saved model:", MODEL_DIR / "car_price_model.joblib")
    print("Saved metrics:", OUTPUT_DIR / "metrics.csv")
    print("Saved plots:", PLOT_DIR)


if __name__ == "__main__":
    main()
