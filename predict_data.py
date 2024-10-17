import datetime
import pandas as pd
import numpy as np
import os
import argparse
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from typing import List, Tuple

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run penalized linear regression with custom data and factor files.')
    parser.add_argument('--data', type=str, default='data.csv', help='Path to the data CSV file')
    parser.add_argument('--factor', type=str, default='factor.csv', help='Path to the factor CSV file')
    parser.add_argument('--work_dir', type=str, default='', help='Working directory (optional)')
    parser.add_argument('--output_dir', type=str, default='', help='Directory to save output files (optional)')
    return parser.parse_args()

def save_file(df, file_name='file.csv', with_index=False):
    df.to_csv(file_name, index=with_index)
    print(f"Saved `{file_name}`.")

def read_file(file_name='file.csv', parse_dates=[]):
    print(f"Read `{file_name}`.")
    return pd.read_csv(file_name, parse_dates=parse_dates)

def inputData(factor_file, data_file):
    factor = list(read_file(factor_file)["variable"].values)
    data = read_file(data_file, parse_dates=['date'])
    return factor, data

def outputData(factor, data, factor_file, data_file):
    save_file(pd.DataFrame({'variable': factor}), factor_file)
    save_file(data, data_file)

def split_data(data: pd.DataFrame, cutoff: List[pd.Timestamp], stock_vars: List[str], ret_var: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    # Vectorized splitting using boolean indexing
    train = data[(data["date"] >= cutoff[0]) & (data["date"] < cutoff[1])]
    test = data[(data["date"] >= cutoff[1]) & (data["date"] < cutoff[2])]

    X_train = train[stock_vars].values
    X_test = test[stock_vars].values

    Y_train = train[ret_var].values
    Y_test = test[ret_var].values

    # Scale the features using RobustScaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Calculate mean of Y_train and create mean-adjusted Y_train_dm
    Y_mean = np.mean(Y_train)
    Y_train_dm = Y_train - Y_mean

    return X_train_scaled, Y_train_dm, X_test_scaled, Y_test, test[["year", "month", "date", "permno", ret_var]]

def train_and_predict(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray) -> dict:
    # Using cross-validated models to find the best alpha automatically
    models = {
        'ols': LinearRegression(),
        'lasso': LassoCV(cv=5),
        'ridge': RidgeCV(cv=5),
        'en': ElasticNetCV(cv=5),  # Automatically tunes alpha and l1_ratio
        'xgb': XGBRegressor()  # Placeholder for XGBoost
    }

    # XGBoost hyperparameter grid
    xgb_params = {
        'n_estimators': [500, 1000],
        'learning_rate': [0.01, 0.1],
        'max_depth': [None],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1]
    }

    # Initialize GridSearchCV for XGBoost
    xgb_model = GridSearchCV(XGBRegressor(objective='reg:squarederror', random_state=42),
                             param_grid=xgb_params, 
                             scoring='neg_mean_squared_error', 
                             cv=TimeSeriesSplit(n_splits=3), 
                             n_jobs=-1)

    # Update the model dictionary
    models['xgb'] = xgb_model

    # Fit all models and predict in a single loop
    predictions = {name: model.fit(X_train, Y_train).predict(X_test) for name, model in models.items()}
    
    return predictions


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    pd.set_option("mode.chained_assignment", None)
    print(datetime.datetime.now())

    # set working directory "Your working directory"
    work_dir = args.work_dir
    output_dir = args.output_dir

    # Output file contains predictions 
    output_path = os.path.join(
        output_dir, "output.csv"
    )  # replace with the correct file name
    
    # read sample data
    data_path = os.path.join(
        work_dir, args.data
    )  # replace with the correct file name

    # read list of predictors for stocks
    factor_path = os.path.join(
        work_dir, args.factor
    )  # replace with the correct file name

    # Assuming inputData returns a tuple of stock_vars and clean data DataFrame
    stock_vars, data = inputData(factor_file=factor_path, data_file=data_path)
    ret_var = "stock_exret"

    starting = pd.to_datetime("20000101", format="%Y%m%d")
    counter = 0
    pred_out = pd.DataFrame()

    start_time = datetime.datetime.now()

    while (starting + pd.DateOffset(years=11 + counter)) <= pd.to_datetime("20240101", format="%Y%m%d"):
        cutoff = [starting + pd.DateOffset(years=i) for i in [0, 10+counter, 11+counter]]
        print(f'[Processing...] Train:{cutoff[0].year}-{cutoff[1].year} | Predict:{cutoff[1].year}-{cutoff[2].year} ', end='')
        
        X_train, Y_train, X_test, Y_test, reg_pred = split_data(data, cutoff, stock_vars, ret_var)
        predictions = train_and_predict(X_train, Y_train, X_test)
        
        for name, pred in predictions.items():
            reg_pred[name] = pred
        
        pred_out = pd.concat([pred_out, reg_pred], ignore_index=True)

        end_time = datetime.datetime.now()
        duration = end_time - start_time
        print(f"| {int(duration.total_seconds() // 60):02}:{int(duration.total_seconds() % 60):02}")
        
        # Go to the next year
        counter += 1

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print(f"Total Time: {int(duration.total_seconds() // 60):02}:{int(duration.total_seconds() % 60):02}")


    save_file(pred_out, output_path)

    yreal = pred_out[ret_var].values
    for model_name in ['ols', 'lasso', 'ridge', 'en', 'xgb']:
        ypred = pred_out[model_name].values
        r2 = r2_score(yreal, ypred)
        print(f"{model_name}: {r2}")

    print(datetime.datetime.now())