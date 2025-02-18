import os 
import sys
repo_dir = "/LLM4RealEstate"
sys.path.append(repo_dir)
import lightgbm
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
from sklearn.metrics import mean_squared_error, mean_absolute_error
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

    lgbm_regr = lightgbm.LGBMRegressor(random_state=seed, verbose=-1)
    lgbm_regr.fit(X_train, y_train)

    cv_mapiets = BlockBootstrap(
        n_resamplings=30, n_blocks=10, overlapping=False, random_state=seed
    )
    mapie_lgbm_regr = MapieTimeSeriesRegressor(lgbm_regr, cv = cv_mapiets, random_state=seed, n_jobs=8)
    mapie_lgbm_regr.fit(X_val, y_val)


    if not os.path.exists(f"{repo_dir}/results"):
        os.makedirs(f"{repo_dir}/results")
    joblib.dump(mapie_lgbm_regr, f"{repo_dir}/results/{dataset}_lgbm.pkl")

    # save feature rankings
    lgbm_explainer = shap.Explainer(mapie_lgbm_regr.estimator)
    shap_values_lgbm = lgbm_explainer(X_test)
    # rank features according to mean absolute shap value
    shap_values_lgbm = np.abs(shap_values_lgbm.values).mean(axis=0)
    shap_values_lgbm = pd.DataFrame(shap_values_lgbm, index=X_test.columns, columns=['shap_value'])
    shap_values_lgbm['feature_importance'] = mapie_lgbm_regr.estimator.feature_importances_
    shap_values_lgbm.to_csv(f"{repo_dir}/results/{dataset}_lgbm_shap.csv")

    train_interval_lgbm = mapie_lgbm_regr.predict(X_train, alpha=0.1)
    val_interval_lgbm = mapie_lgbm_regr.predict(X_val, alpha=0.1)  
    test_interval_lgbm = mapie_lgbm_regr.predict(X_test, alpha=0.1)

    train_coverage = regression_coverage_score_v2(y_train, train_interval_lgbm[1])[0]
    val_coverage = regression_coverage_score_v2(y_val, val_interval_lgbm[1])[0]
    test_coverage = regression_coverage_score_v2(y_test, test_interval_lgbm[1])[0]
    train_width = regression_mean_width_score(train_interval_lgbm[1][:,0], train_interval_lgbm[1][:,1])
    val_width = regression_mean_width_score(val_interval_lgbm[1][:,0], val_interval_lgbm[1][:,1])
    test_width = regression_mean_width_score(test_interval_lgbm[1][:,0], test_interval_lgbm[1][:,1])

    train_metrics = {
        "coverage": train_coverage,
        "width": train_width,
        "rmse": np.sqrt(mean_squared_error(y_train,train_interval_lgbm[0])),
        "mae": mean_absolute_error(y_train,train_interval_lgbm[0])
    }

    val_metrics = {
        "coverage": val_coverage,
        "width": val_width,
        "rmse": np.sqrt(mean_squared_error(y_val,val_interval_lgbm[0])),
        "mae": mean_absolute_error(y_val,val_interval_lgbm[0])
    }

    test_metrics = {
        "coverage": test_coverage,
        "width": test_width,
        "rmse": np.sqrt(mean_squared_error(y_test,test_interval_lgbm[0])),
        "mae": mean_absolute_error(y_test,test_interval_lgbm[0])
    }

    metrics = pd.DataFrame([train_metrics, test_metrics], index=['train', 'test'])
    metrics.to_csv(f"{repo_dir}/results/{dataset}_lgbm.csv")

    # save predictions
    preds = pd.DataFrame({
        'y_true': y_test,
        'y_pred': test_interval_lgbm[0],
        'lower': test_interval_lgbm[1][:,0][:,0],
        'upper': test_interval_lgbm[1][:,1][:,0],
        'index': test.index
    })

    preds.to_csv(f"{repo_dir}/results/{dataset}_lgbm_preds.csv")


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

    metrics.to_csv(f"{repo_dir}/results/{dataset}_knn.csv")

    # save predictions
    preds = pd.DataFrame({
        'y_true': y_test,
        'knn_3_geo': knn_3_geo_test_preds,
        'knn_3_hedonic': knn_3_hedonic_test_preds,
        'knn_10_geo': knn_10_geo_test_preds,
        'knn_10_hedonic': knn_10_hedonic_test_preds,
        'index': test.index
    })
    preds.to_csv(f"{repo_dir}/results/{dataset}_knn_preds.csv")
    




if __name__ == "__main__":
    args = parser.parse_args()
    run_baselines(args.dataset)
