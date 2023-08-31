import numpy as np
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR

app = Flask(__name__)

# Initialize the car_prices dictionary
car_prices = {
    2022: 0,
    2023: 0,
}

# Create regression models with default hyperparameters
linear_regression = LinearRegression()
ridge_regression = Ridge()
lasso_regression = Lasso()
decision_tree_regression = DecisionTreeRegressor()
random_forest_regression = RandomForestRegressor()
svr_regression = SVR()

# Create a VotingRegressor with all the regression models
voting_regressor = VotingRegressor([
    ('linear', linear_regression),
    ('ridge', ridge_regression),
    ('lasso', lasso_regression),
    ('decision_tree', decision_tree_regression),
    ('random_forest', random_forest_regression),
    ('svr', svr_regression)
])

@app.route("/", methods=["GET", "POST"])
def predict_car_price():
    predicted_price = None  # Initialize to None

    if request.method == "POST":
        car_prices[2022] = float(request.form["year_2022"])
        car_prices[2023] = float(request.form["year_2023"])

        # Prepare the data for scikit-learn
        years = np.array(list(car_prices.keys())).reshape(-1, 1)
        prices = np.array(list(car_prices.values()))

        # Train the VotingRegressor on the historical data
        voting_regressor.fit(years, prices)

        # Collect input from the user for the year to predict
        prediction_year = int(request.form["prediction_year"])

        # Predict car price for the specified future year using the VotingRegressor
        predicted_price = voting_regressor.predict([[prediction_year]])
        predicted_price = predicted_price[0]

    return render_template("predict.html", predicted_price=predicted_price)

if __name__ == "__main__":
    app.run(debug=True)
