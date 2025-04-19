'''
Author: JYYD jyyd23@mails.tsinghua.edu.cn
Date: 2024-07-11 13:27:14
LastEditors: JYYD jyyd23@mails.tsinghua.edu.cn
LastEditTime: 2024-07-12 22:58:50
FilePath: \PNC\pm_code\stack_cross.py
Description: 

'''
import os, sys
os.chdir(sys.path[0])
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
import time
from sklearn.model_selection import GridSearchCV, cross_val_score,cross_val_predict,cross_validate, RandomizedSearchCV
import pylab
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, Lasso
from sklearn import neighbors, svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, StackingRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from joblib import dump, load 
      


def load_feature_data(del_station:str, workstation_Flag: bool = False):
    # load the data
    if workstation_Flag:
        pollution_data = pd.read_csv('../code/dataset/NABEL/feature_data/feature_data_PM_all.csv')
    else:
        pollution_data = pd.read_csv('../code/pncEstimator-main/data/NABEL/feature_eng/feature_data_PM_all.csv')
    pollution_data = pollution_data.dropna()  # drop the rows with missing values
    pollution_data['Date/time'] = pd.to_datetime(pollution_data['Date/time'])  # convert the date column to datetime
    pollution_data = pollution_data[(pollution_data['Date/time'] >= '2016-01-01 01:00')]
    
    pollution_data_val = pollution_data[(pollution_data['Date/time']>='2016-01-01 01:00') &
                                        (pollution_data['Date/time']<'2017-01-01 01:00')]
    pollution_data_val  = pollution_data_val .reset_index(drop=True)
    pollution_data_test = pollution_data[(pollution_data['Date/time']>='2021-01-01 01:00') &
                                        (pollution_data['Date/time']<'2022-01-01 01:00')]
    pollution_data_test = pollution_data_test.reset_index(drop=True)
    
    station_list = pollution_data['station'].unique()

    pollution_data_filtered = pollution_data[pollution_data['station'] != del_station]
    pollution_data_train_filtered = pollution_data_filtered[(pollution_data_filtered['Date/time'] >= '2017-01-01 01:00') &
                                                    (pollution_data_filtered['Date/time'] < '2021-01-01 01:00')]
    pollution_data_train_filtered = pollution_data_train_filtered.reset_index(drop=True)

    return station_list, pollution_data_train_filtered, pollution_data_test, pollution_data_val


def pollution_data_split(pollution_type:str, pollution_data_type):
    pollution_data_type = pollution_data_type.replace('-', np.nan)
    pollution_data_type = pollution_data_type.fillna(0)

    pollution_type = [pollution_type]

    columns = ['Date/time', 'station'] + pollution_type + [
        'pm25_cams', 'pm10_cams','no2_cams', 'o3_cams',
        'Radiation[W/m2] meteo', 'Temperature meteo',
        'Precipitation[mm] meteo', 'Relative humidity[%] meteo',
        'Wind speed[m/s] meteo', 'trafficVol', 'hour', 'month', 'weekday'
    ]

    pollution_data_type = pollution_data_type[columns]
    pollution_x = pollution_data_type.iloc[:,3:].values
    pollution_y = pollution_data_type.iloc[:,2:3].values
    
    return pollution_x, pollution_y

def scaler_pollution(del_station, pollution_type:str, workstation_Flag: bool = False):
    station_list, pollution_data_train_filtered, pollution_data_test, pollution_data_val = load_feature_data(del_station, workstation_Flag)
    scaler_std = StandardScaler()
    pollution_x_train, pollution_y_train = pollution_data_split(pollution_type, pollution_data_train_filtered)
    pollution_x_test, pollution_y_test = pollution_data_split(pollution_type, pollution_data_test)
    pollution_x_val, pollution_y_val = pollution_data_split(pollution_type, pollution_data_val)
    pollution_x_all = np.vstack((pollution_x_train, pollution_x_test, pollution_x_val))
    pollution_x_all_scaler = scaler_std.fit(pollution_x_all)
    scaler_x_train = pollution_x_all_scaler.transform(pollution_x_train)
    scaler_x_test = pollution_x_all_scaler.transform(pollution_x_test)
    scaler_x_val = pollution_x_all_scaler.transform(pollution_x_val)

    return station_list, scaler_x_train, pollution_y_train, scaler_x_test, pollution_y_test, scaler_x_val, pollution_y_val

# evaluation
def pred_plot(y_test, y_pred):
    print('----------------------------------')
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    print('MSE: ', mse)
    print('MAE: ', mae)
    print('r2 score: ', r2)
    print('Explained_variance: ', evs)
    return mse,mae,r2,evs
def predpnc(model_fit, x_test, y_test, pncdata2020):
    pred = model_fit.predict(x_test)
    mse, mae, r2, evs = pred_plot(y_test, pred)
    pred = pd.DataFrame(pred)
    data2020_pred = pd.concat([pncdata2020, pred], axis=1)
    return data2020_pred


def storage_parameters(pollution_type_full):
    pollution_types_short = ['pm25', 'pm10', 'nox', 'no2', 'o3']
    pollution_types_full = ['PM2.5 [ug/m3]', 'PM10 [ug/m3]', 'NOX [ug/m3 eq. NO2]',
                            'NO2 [ug/m3]', 'O3 [ug/m3]']
    model_params = {
        'lasso_alpha': [0.1, 1, 0.01, 0.5, 0.01],
        'knn_n_neighbors': [50, 50, 100, 100, 50],
        'knn_weights': ['distance', 'distance', 'distance', 'uniform', 'distance'],
        'dt_criterion': ['poisson', 'poisson', 'poisson', 'absolute_error', 'absolute_error'],
        'dt_max_depth': [60, 60, 60, 30, 60],
        'rf_n_estimators': [100, 100, 100, 100, 100],
        'rf_max_depth': [None, 20, 20, 10, 5],
        'ada_learning_rate': [0.01, 0.01, 0.5, 0.01, 0.01],
        'ada_n_estimators': [200, 200, 100, 100, 200],
        'gbr_learning_rate': [0.05, 0.01, 0.1, 0.1, 0.05],
        'gbr_max_depth': [5, 5, 3, 3, 5],
        'gbr_n_estimators': [100, 100, 100, 100, 100],
        'lgb_learning_rate': [0.1, 0.1, 0.05, 0.1, 0.1],
        'lgb_n_estimators': [100, 100, 100, 100, 100],
        'xgb_n_estimators': [100, 100, 200, 50, 200],
        'xgb_learning_rate': [0.1, 0.1, 0.05, 0.1, 0.1]
    }

    if pollution_type_full not in pollution_types_full:
        raise ValueError(f"Invalid pollution type: {pollution_type_full}")

    i = pollution_types_full.index(pollution_type_full)
    params = {param: values[i] for param, values in model_params.items()}
    params['short_name'] = pollution_types_short[i]

    return params


def load_model(pollution_type_short ):
    if pollution_type_short  == 'pm25':
        pm25_regr = load('./model/pollution/stack_trainedModel_pm25.joblib')
        return pm25_regr
    elif pollution_type_short  == 'pm10':
        pm10_regr = load('./model/pollution/stack_trainedModel_pm10.joblib')
        return pm10_regr
    elif pollution_type_short  == 'nox':
        nox_regr = load('./model/pollution/stack_trainedModel_nox.joblib')
        return nox_regr
    elif pollution_type_short  == 'no2':
        no2_regr = load('./model/pollution/stack_trainedModel_no2.joblib')
        return no2_regr
    elif pollution_type_short  == 'o3':
        o3_regr = load('./model/pollution/stack_trainedModel_o3.joblib')
        return o3_regr

def main(workstation_Flag: bool = False):
    pollution_types = ['PM2.5 [ug/m3]', 'PM10 [ug/m3]', 'NOX [ug/m3 eq. NO2]',
                    'NO2 [ug/m3]', 'O3 [ug/m3]']
    for pollution_type in pollution_types:
        station_list, pollution_data_train_filtered, pollution_data_test, _ = load_feature_data(workstation_Flag)
        for del_station in station_list:
            print(f"Training model for {pollution_type} without station: {del_station}")
            _, pollution_x_train, pollution_y_train, pollution_x_test, pollution_y_test, _, _ = scaler_pollution(del_station, pollution_type, workstation_Flag)
            params = storage_parameters(pollution_type)
            pollution_type_short = params.pop('short_name')
            pollution_stack_regr = load_model(pollution_type_short)
            pollution_stack_cv_fit = pollution_stack_regr.fit(pollution_x_train, pollution_y_train)
            pollution_stack_cv_pred = pollution_stack_regr.predict(pollution_x_test)
            stack_mse, stack_mae, stack_r2, stack_evs = pred_plot(pollution_y_test, pollution_stack_cv_pred)
            data2020_pred11 = predpnc(pollution_stack_cv_fit, pollution_x_test,
                                      pollution_y_test, pollution_data_test)
            new_columns = [f'stack_pred_{del_station}']
            data2020_pred11.columns = list(pollution_data_test.columns) + new_columns
            data2020_pred11.to_csv(f'./out/pollution/cv/{pollution_type_short}_data_pred_stack_{del_station}.csv')

            # Save train set predictions
            train_predictions = pollution_stack_cv_fit.predict(pollution_x_train)
            train_results = pd.DataFrame(pollution_y_train, columns=['Actual'])
            train_results[f'stack_pred_{del_station}'] = train_predictions
            train_results.to_csv(f'./out/pollution/cv/del/{pollution_type_short}_train_results_stack_{del_station}.csv', index=False)

            # Calculate and print train metrics
            train_r2 = r2_score(pollution_y_train, train_predictions)
            train_rmse = mean_squared_error(pollution_y_train, train_predictions, squared=False)
            train_mae = mean_absolute_error(pollution_y_train, train_predictions)
            print(f"{pollution_type}--{del_station} - Train R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
        
        print(f"{pollution_type}--{del_station} - MSE: {stack_mse}, MAE: {stack_mae}, R2: {stack_r2}, EVS: {stack_evs}")

if __name__ == '__main__':
    workstation_Flag = True
    main(workstation_Flag)