import os 
import sys
repo_dir = "/LLM4RealEstate"
sys.path.append(repo_dir)
import lightgbm
import xgboost
import pandas as pd
import numpy as np
import json
from sklearn.neighbors import KNeighborsRegressor
import argparse
import joblib
from mapie.regression import MapieTimeSeriesRegressor
from mapie.subsample import BlockBootstrap
from mapie.metrics import regression_coverage_score_v2, regression_mean_width_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import randint, uniform
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import shap

seed = 0

parser = argparse.ArgumentParser(description='Run baselines')
parser.add_argument(
                    "--dataset",
                    action="store",
                    type=str,
                    default='KC',
                    choices=['KC', 'beijing', 'barcelona'],
                    help="Give the dataset you want to use",
)

def run_baselines(dataset):
    # Load the data
    df = pd.read_csv(f'data/processed/{dataset}.csv', index_col=0)

    df.transaction_date = pd.to_datetime(df.transaction_date)

    train = df[df['set'] == 'train']
    val = df[df['set'] == 'val']
    test = df[df['set'] == 'test']
    
    features = json.load(open(f"{repo_dir}/config/features.json", 'r'))[dataset]
    test_indices = np.loadtxt(f"{repo_dir}/config/{dataset}_test_indices.txt")

    X_train = train[features['hedonic']+['transaction_date']]
    start_date = X_train.transaction_date.min()
    X_train['transaction_date'] = (X_train.transaction_date - start_date).dt.days
    y_train = train['price'] if 'price' in train.columns else np.exp(train['log_price'])

    X_val = val[features['hedonic']+['transaction_date']]
    X_val['transaction_date'] = (X_val.transaction_date - start_date).dt.days
    y_val = val['price'] if 'price' in val.columns else np.exp(val['log_price'])

    X_test = test[features['hedonic']+['transaction_date']]
    X_test['transaction_date'] = (X_test.transaction_date - start_date).dt.days
    y_test = test['price'] if 'price' in test.columns else np.exp(test['log_price'])

    lgbm_regr = lightgbm.LGBMRegressor(random_state=seed, verbose=-1)
    lgbm_regr.fit(X_train, y_train)
    train_pred = lgbm_regr.predict(X_train)
    val_pred = lgbm_regr.predict(X_val)
    test_pred = lgbm_regr.predict(X_test)

    train_metrics = {
        "rmse": np.sqrt(mean_squared_error(y_train, train_pred)),
        "mae": mean_absolute_error(y_train,train_pred),
        "mape": mean_absolute_percentage_error(y_train,train_pred)
    }

    val_metrics = {
        "rmse": np.sqrt(mean_squared_error(y_val,val_pred)),
        "mae": mean_absolute_error(y_val,val_pred),
        "mape": mean_absolute_percentage_error(y_val,val_pred)
    }

    test_metrics = {
        "rmse": np.sqrt(mean_squared_error(y_test,test_pred)),
        "mae": mean_absolute_error(y_test,test_pred),
        "mape": mean_absolute_percentage_error(y_test,test_pred)
    }

    metrics = pd.DataFrame([train_metrics, test_metrics], index=['train', 'test'])
    # if results folder does not exist, create it
    if not os.path.exists(f"{repo_dir}/results"):
        os.makedirs(f"{repo_dir}/results")
    metrics.to_csv(f"{repo_dir}/results/{dataset}_lgbm_hedonic.csv")

    # save predictions
    preds = pd.DataFrame({
        'y_true': y_test,
        'y_pred': test_pred,
        'index': test.index
    })

    preds.to_csv(f"{repo_dir}/results/{dataset}_lgbm_hedonic_preds.csv")
    




if __name__ == "__main__":
    args = parser.parse_args()
    run_baselines(args.dataset)
