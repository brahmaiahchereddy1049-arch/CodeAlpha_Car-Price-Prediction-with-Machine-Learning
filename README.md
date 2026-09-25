# Car Price Prediction using Machine Learning

A complete machine learning project that predicts the selling price of used cars using vehicle attributes such as present price, manufacturing year, kilometers driven, fuel type, selling type, transmission, and previous owners.

## Project Overview

Used-car prices depend on several factors including the original/current price, vehicle age, mileage, fuel type, transmission, and ownership history.

This project builds a regression-based machine learning pipeline to:

- Load and inspect car data
- Clean and preprocess the dataset
- Perform feature engineering
- Encode categorical variables
- Train multiple regression models
- Compare models using MAE, RMSE, and R²
- Visualize the results
- Save the best model
- Predict the price of a new car

## Dataset

The project uses `data/car_data.csv`.

### Features

| Feature | Description |
|---|---|
| `Car_Name` | Name/model of the car |
| `Year` | Manufacturing year |
| `Selling_Price` | Target variable; selling price in lakh |
| `Present_Price` | Current/ex-showroom price in lakh |
| `Driven_kms` | Kilometers driven |
| `Fuel_Type` | Petrol, Diesel, or CNG |
| `Selling_type` | Dealer or Individual |
| `Transmission` | Manual or Automatic |
| `Owner` | Number of previous owners |

### Engineered Feature

`Car_Age = current year - Year`

The `Car_Name` column is not used as a model feature because it can create many high-cardinality categories and is not necessary for the baseline prediction model.

## Machine Learning Models

The project compares:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

### Evaluation Metrics

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of Determination

The best model is selected using the highest R² score on the held-out test set.

> Note: Model results depend on the train/test split and random seed. They should not be interpreted as a guarantee of real-world performance.

## Project Structure

```text
car_price_prediction_ml/
│
├── data/
│   └── car_data.csv
│
├── models/
│   └── car_price_model.joblib       # created after training
│
├── outputs/
│   ├── metrics.csv                  # created after training
│   └── plots/
│       ├── actual_vs_predicted.png
│       ├── price_distribution.png
│       └── present_vs_selling_price.png
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/car-price-prediction-ml.git
cd car-price-prediction-ml
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Train the Models

Run:

```bash
python src/train.py
```

The script will:

1. Load the CSV file
2. Remove duplicate rows
3. Check missing values
4. Create the `Car_Age` feature
5. Split the data into training and testing sets
6. Encode categorical variables
7. Train three regression models
8. Calculate MAE, RMSE, and R²
9. Save the metrics
10. Save the best model
11. Generate visualization plots

Generated files:

```text
models/car_price_model.joblib
outputs/metrics.csv
outputs/plots/actual_vs_predicted.png
outputs/plots/price_distribution.png
outputs/plots/present_vs_selling_price.png
```

## Predict a New Car Price

After training, run:

```bash
python src/predict.py
```

The script uses an example vehicle:

```text
Present Price: 8.5 lakh
Driven KMs: 35000
Car Age: 5 years
Fuel Type: Petrol
Selling Type: Dealer
Transmission: Manual
Previous Owners: 0
```

It then prints the predicted selling price.

## Example Output

```text
Best model: Decision Tree
Predicted selling price: 6.42 lakh
```

The exact result can change if the dataset, model parameters, or train/test split changes.

## Workflow

```text
Raw Dataset
     |
     v
Data Cleaning
     |
     v
Feature Engineering
     |
     v
Categorical Encoding
     |
     v
Train/Test Split
     |
     v
Model Training
     |
     +-------------------+
     |        |          |
     v        v          v
 Linear   Decision    Random
Regression  Tree      Forest
     |        |          |
     +--------+----------+
              |
              v
       Model Evaluation
              |
              v
       Best Model Saved
              |
              v
       Price Prediction
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

## Real-World Applications

Car price prediction can be used by:

- Used-car marketplaces
- Dealerships
- Vehicle owners
- Automobile valuation platforms
- Financial institutions
- Pricing and inventory teams

## Limitations

This is an educational machine learning project. Real-world vehicle valuation may require additional variables such as:

- Exact vehicle variant
- Location
- Accident history
- Service history
- Insurance status
- Number of previous accidents
- Vehicle condition
- Market demand
- Seasonal effects

The dataset is relatively small, so the model should be validated on larger and more representative data before production use.

## Future Improvements

- Add XGBoost/Gradient Boosting models
- Perform hyperparameter tuning with GridSearchCV or RandomizedSearchCV
- Add cross-validation
- Add more vehicle-condition features
- Build a Streamlit web application
- Deploy the model using a cloud platform
- Add SHAP-based model explainability

## Author

**Brahmaiah Chereddy**

Data Analyst / Machine Learning Project

GitHub: `https://github.com/brahmaiahchereddy1049-arch`

LinkedIn: `https://www.linkedin.com/in/brahmaiah-chereddy-5b0050346/`

## License

This project is intended for educational and portfolio purposes. Check the original dataset's license/terms before redistributing the dataset publicly.
