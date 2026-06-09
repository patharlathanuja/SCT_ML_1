import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def load_data() -> pd.DataFrame:
    # Example dataset: square footage, bedrooms, bathrooms, and price
    data = {
        "square_feet": [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700],
        "bedrooms": [3, 3, 3, 4, 2, 3, 4, 4, 3, 3],
        "bathrooms": [2, 2, 2, 2, 1, 2, 3, 2, 2, 2],
        "price": [245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000],
    }
    return pd.DataFrame(data)


def train_linear_regression(df: pd.DataFrame):
    X = df[["square_feet", "bedrooms", "bathrooms"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return model, X_test, y_test, y_pred, rmse, r2


def print_metrics(model, rmse: float, r2: float):
    print("Linear Regression model trained")
    print("Intercept:", model.intercept_)
    print("Coefficients:")
    print("  square_feet:", model.coef_[0])
    print("  bedrooms:", model.coef_[1])
    print("  bathrooms:", model.coef_[2])
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 score: {r2:.4f}")


def predict_price(model, square_feet: float, bedrooms: int, bathrooms: int) -> float:
    features = np.array([[square_feet, bedrooms, bathrooms]])
    return float(model.predict(features)[0])


def main():
    df = load_data()
    model, X_test, y_test, y_pred, rmse, r2 = train_linear_regression(df)
    print_metrics(model, rmse, r2)

    sample_house = {
        "square_feet": 1800,
        "bedrooms": 3,
        "bathrooms": 2,
    }
    predicted_price = predict_price(
        model,
        sample_house["square_feet"],
        sample_house["bedrooms"],
        sample_house["bathrooms"],
    )

    print("\nSample prediction:")
    print(
        f"House with {sample_house['square_feet']} sq ft, "
        f"{sample_house['bedrooms']} bedrooms, "
        f"{sample_house['bathrooms']} bathrooms -> predicted price ${predicted_price:,.0f}"
    )

    print("\nTest samples:")
    for actual, pred in zip(y_test.values, y_pred):
        print(f"  actual ${actual:,.0f}, predicted ${pred:,.0f}")


if __name__ == "__main__":
    main()
