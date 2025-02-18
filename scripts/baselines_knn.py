import os 
import sys
repo_dir = "/LLM4RealEstate"
sys.path.append(repo_dir)
import lightgbm
import xgboost
import pandas as pd
import numpy as np
import json
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
import argparse
import joblib
from mapie.regression import MapieTimeSeriesRegressor
from mapie.subsample import BlockBootstrap
from mapie.metrics import regression_coverage_score_v2, regression_mean_width_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import randint, uniform
from sklearn.metrics import mean_squared_error, mean_absolute_error
import shap

seed = 0

parser = argparse.ArgumentParser(description='Run')
parser.add_argument(
                    "--dataset",
                    action="store",
                    type=str,
                    default='KC',
                    choices=['KC', 'beijing', 'barcelona'],
                    help="Give the dataset you want to use",
)

def run(dataset):
    # Load the data
    df = pd.read_csv(f'data/processed/{dataset}.csv', index_col=0)

    df.transaction_date = pd.to_datetime(df.transaction_date)

    train = df[df['set'] == 'train']
    val = df[df['set'] == 'val']
    test = df[df['set'] == 'test']
    
    features = json.load(open(f"{repo_dir}/config/features.json", 'r'))[dataset]
    test_indices = np.loadtxt(f"{repo_dir}/config/{dataset}_test_indices.txt")

    X_train = train[features['hedonic']+['x_geo', 'y_geo', 'transaction_date']]
    start_date = X_train.transaction_date.min()
    X_train['transaction_date'] = (X_train.transaction_date - start_date).dt.days
    y_train = train['price'] if 'price' in train.columns else np.exp(train['log_price'])

    X_val = val[features['hedonic']+['x_geo', 'y_geo', 'transaction_date']]
    X_val['transaction_date'] = (X_val.transaction_date - start_date).dt.days
    y_val = val['price'] if 'price' in val.columns else np.exp(val['log_price'])

    X_test = test[features['hedonic']+['x_geo', 'y_geo', 'transaction_date']]
    X_test['transaction_date'] = (X_test.transaction_date - start_date).dt.days
    y_test = test['price'] if 'price' in test.columns else np.exp(test['log_price'])

    # train knn
    # 1. KNN k=3, geo
    knn_3_geo = KNeighborsRegressor(n_neighbors=3, metric='haversine')
    knn_3_geo.fit(X_train[['x_geo', 'y_geo']], y_train)
    knn_3_geo_train_preds = knn_3_geo.predict(X_train[['x_geo', 'y_geo']])
    knn_3_geo_test_preds = knn_3_geo.predict(X_test[['x_geo', 'y_geo']])

    # 2. KNN k=3, hedonic
    knn_3_hedonic = KNeighborsRegressor(n_neighbors=3, metric="cosine")
    knn_3_hedonic.fit(X_train[features['hedonic']], y_train)
    knn_3_hedonic_train_preds = knn_3_hedonic.predict(X_train[features['hedonic']])
    knn_3_hedonic_test_preds = knn_3_hedonic.predict(X_test[features['hedonic']])

    # 3. KNN k=10, geo
    knn_10_geo = KNeighborsRegressor(n_neighbors=10, metric='haversine')
    knn_10_geo.fit(X_train[['x_geo', 'y_geo']], y_train)
    knn_10_geo_train_preds = knn_10_geo.predict(X_train[['x_geo', 'y_geo']])
    knn_10_geo_test_preds = knn_10_geo.predict(X_test[['x_geo', 'y_geo']])

    # 4. KNN k=10, hedonic
    knn_10_hedonic = KNeighborsRegressor(n_neighbors=10, metric="cosine")
    knn_10_hedonic.fit(X_train[features['hedonic']], y_train)
    knn_10_hedonic_train_preds = knn_10_hedonic.predict(X_train[features['hedonic']])
    knn_10_hedonic_test_preds = knn_10_hedonic.predict(X_test[features['hedonic']])

    # Check BallTree equivalence
    # k=10, geo
    knn_10_geo_bt = NearestNeighbors(n_neighbors=10, algorithm='ball_tree', metric='haversine')
    knn_10_geo_bt.fit(X_train[['x_geo', 'y_geo']])
    knn_10_geo_bt_test_neighbors = knn_10_geo_bt.kneighbors(X_test[['x_geo', 'y_geo']])[1]
    print(knn_10_geo_bt_test_neighbors.shape)
    knn_10_geo_bt_test_neighbor_prices = np.array([y_train.iloc[neighbors] for neighbors in knn_10_geo_bt_test_neighbors])
    print(knn_10_geo_bt_test_neighbor_prices.shape)
    knn_10_geo_bt_test_preds = np.mean(knn_10_geo_bt_test_neighbor_prices, axis=1)
    assert np.all(knn_10_geo_test_preds == knn_10_geo_bt_test_preds)

    # 5. KNN k=10, mixed
    example_file = f'config/{dataset}_test_10_mixed_examples.json'
    if os.path.exists(example_file):
        with open(example_file) as f:
            neighbors = json.load(f)
    knn_10_geo_test_neighbors = np.array([y_train.iloc[neighbors[str(int(test_id))]] for test_id in test_indices])
    knn_10_mixed_test_preds = knn_10_geo_test_neighbors.mean(axis=1)

    
    index = pd.MultiIndex.from_product([['knn_3_geo', 'knn_3_hedonic', 'knn_10_geo', 'knn_10_hedonic'], ['train', 'test']], names=['model', 'set'])
    metrics = pd.DataFrame(columns=['rmse', 'mae'], index=index)
    metrics.loc[('knn_3_geo', 'train'), 'rmse'] = np.sqrt(mean_squared_error(y_train, knn_3_geo_train_preds))
    metrics.loc[('knn_3_geo', 'train'), 'mae'] = mean_absolute_error(y_train, knn_3_geo_train_preds)
    metrics.loc[('knn_3_geo', 'test'), 'rmse'] = np.sqrt(mean_squared_error(y_test, knn_3_geo_test_preds))
    metrics.loc[('knn_3_geo', 'test'), 'mae'] = mean_absolute_error(y_test, knn_3_geo_test_preds)
    metrics.loc[('knn_3_hedonic', 'train'), 'rmse'] = np.sqrt(mean_squared_error(y_train, knn_3_hedonic_train_preds))
    metrics.loc[('knn_3_hedonic', 'train'), 'mae'] = mean_absolute_error(y_train, knn_3_hedonic_train_preds)
    metrics.loc[('knn_3_hedonic', 'test'), 'rmse'] = np.sqrt(mean_squared_error(y_test, knn_3_hedonic_test_preds))
    metrics.loc[('knn_3_hedonic', 'test'), 'mae'] = mean_absolute_error(y_test, knn_3_hedonic_test_preds)
    metrics.loc[('knn_10_geo', 'train'), 'rmse'] = np.sqrt(mean_squared_error(y_train, knn_10_geo_train_preds))
    metrics.loc[('knn_10_geo', 'train'), 'mae'] = mean_absolute_error(y_train, knn_10_geo_train_preds)
    metrics.loc[('knn_10_geo', 'test'), 'rmse'] = np.sqrt(mean_squared_error(y_test, knn_10_geo_test_preds))
    metrics.loc[('knn_10_geo', 'test'), 'mae'] = mean_absolute_error(y_test, knn_10_geo_test_preds)
    metrics.loc[('knn_10_hedonic', 'train'), 'rmse'] = np.sqrt(mean_squared_error(y_train, knn_10_hedonic_train_preds))
    metrics.loc[('knn_10_hedonic', 'train'), 'mae'] = mean_absolute_error(y_train, knn_10_hedonic_train_preds)
    metrics.loc[('knn_10_hedonic', 'test'), 'rmse'] = np.sqrt(mean_squared_error(y_test, knn_10_hedonic_test_preds))
    metrics.loc[('knn_10_hedonic', 'test'), 'mae'] = mean_absolute_error(y_test, knn_10_hedonic_test_preds)
    metrics.loc[('knn_10_mixed', 'test'), 'rmse'] = np.sqrt(mean_squared_error(y_test.loc[test_indices], knn_10_mixed_test_preds))
    metrics.loc[('knn_10_mixed', 'test'), 'mae'] = mean_absolute_error(y_test.loc[test_indices], knn_10_mixed_test_preds)

    if not os.path.exists(f"{repo_dir}/results"):
        os.makedirs(f"{repo_dir}/results")
    metrics.to_csv(f"{repo_dir}/results/{dataset}_knn.csv")

    # save predictions
    preds = pd.DataFrame({
        'y_true': y_test,
        'knn_3_geo': knn_3_geo_test_preds,
        'knn_3_hedonic': knn_3_hedonic_test_preds,
        'knn_10_geo': knn_10_geo_test_preds,
        'knn_10_hedonic': knn_10_hedonic_test_preds,
    }, index=test.index)
    knn_10_mixed_df = pd.DataFrame({
        'knn_10_mixed': knn_10_mixed_test_preds
    }, index=test_indices)
    preds = pd.concat([preds, knn_10_mixed_df], axis=1)

    preds.to_csv(f"{repo_dir}/results/{dataset}_knn_preds.csv")
    




if __name__ == "__main__":
    args = parser.parse_args()
    run(args.dataset)
