"""
Predict the selling price of a car using the trained model.

Run from the project root:
    python src/predict.py
"""

from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "car_price_model.joblib"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python src/train.py` first."
        )

    model = joblib.load(MODEL_PATH)

    # Change these values to predict another car.
    car = pd.DataFrame(
        [{
            "Present_Price": 8.5,
            "Driven_kms": 35000,
            "Car_Age": 5,
            "Fuel_Type": "Petrol",
            "Selling_type": "Dealer",
            "Transmission": "Manual",
            "Owner": 0,
        }]
    )

    prediction = model.predict(car)[0]

    print("Car Price Prediction")
    print("-" * 30)
    print(f"Predicted selling price: {prediction:.2f} lakh")


if __name__ == "__main__":
    main()
