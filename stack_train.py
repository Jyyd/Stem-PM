'''
Author: JYYD jyyd23@mails.tsinghua.edu.cn
Date: 2024-07-10 21:29:59
LastEditors: JYYD jyyd23@mails.tsinghua.edu.cn
LastEditTime: 2024-07-12 16:04:34
FilePath: \PNC\pm_code\stack_train.py
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

# prepare the data
def load_feature_data(workstation_Flag: bool = False):
    # load the data
    if workstation_Flag:
        pollution_data = pd.read_csv('../code/dataset/NABEL/feature_data/feature_data_PM_all.csv')
    else:
        pollution_data = pd.read_csv('../code/pncEstimator-main/data/NABEL/feature_eng/feature_data_PM_all.csv')
    pollution_data = pollution_data.dropna() # drop the rows with missing values
    pollution_data['Date/time'] = pd.to_datetime(pollution_data['Date/time']) # convert the date column to datetime

    pollution_data = pollution_data[(pollution_data['Date/time']>='2016-01-01 01:00')]

    pollution_data_val = pollution_data[(pollution_data['Date/time']>='2016-01-01 01:00') &
                                        (pollution_data['Date/time']<'2017-01-01 01:00')]
    pollution_data_val  = pollution_data_val .reset_index(drop=True)

    pollution_data_train = pollution_data[(pollution_data['Date/time']>='2017-01-01 01:00') &
                                        (pollution_data['Date/time']<'2021-01-01 01:00')]
    pollution_data_train = pollution_data_train.reset_index(drop=True)
    
    pollution_data_test = pollution_data[(pollution_data['Date/time']>='2021-01-01 01:00') &
                                        (pollution_data['Date/time']<'2022-01-01 01:00')]
    pollution_data_test = pollution_data_test.reset_index(drop=True)

    print('------------------station num------------------')
    print('the station contains:', len(pollution_data['station'].unique()),
        pollution_data['station'].unique())
    # print('pollution data columns: ', pollution_data.columns)
    print('train length: ', len(pollution_data_train)/(len(pollution_data_train)+len(pollution_data_test)+len(pollution_data_val)),
        len(pollution_data_train))
    print('test length: ', len(pollution_data_test)/(len(pollution_data_train)+len(pollution_data_test)+len(pollution_data_val)),
        len(pollution_data_test))
    print('val length: ', len(pollution_data_val)/(len(pollution_data_train)+len(pollution_data_test)+len(pollution_data_val)),
        len(pollution_data_val))
    return pollution_data_train, pollution_data_test, pollution_data_val
    

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

def scaler_pollution(pollution_type:str, workstation_Flag: bool = False):
    pollution_data_train, pollution_data_test, pollution_data_val = load_feature_data(workstation_Flag)
    scaler_std = StandardScaler()
    pollution_x_train, pollution_y_train = pollution_data_split(pollution_type, pollution_data_train)
    pollution_x_test, pollution_y_test = pollution_data_split(pollution_type, pollution_data_test)
    pollution_x_val, pollution_y_val = pollution_data_split(pollution_type, pollution_data_val)
    # pollution_x_all = np.vstack((pollution_x_train, pollution_x_test, pollution_x_val))
    pollution_x_scaler = scaler_std.fit(pollution_x_train)
    scaler_x_train = pollution_x_scaler.transform(pollution_x_train)
    scaler_x_test = pollution_x_scaler.transform(pollution_x_test)
    scaler_x_val = pollution_x_scaler.transform(pollution_x_val)

    return scaler_x_train, pollution_y_train, scaler_x_test, pollution_y_test, scaler_x_val, pollution_y_val

# evaluation
def pred_plot(y_test, y_pred):
    print('----------------pred metrics------------------')
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


class ModelGridSearchCV:
    def __init__(self, val_x, val_y, pollution_type):
        self.val_x = val_x
        self.val_y = val_y
        self.pollution_type = pollution_type

    def lasso_grid_cv(self):
        param_grid = [{'alpha': [0.01, 0.1, 0.5, 1, 10, 25, 50, 80, 100, 500, 1000]}]
        lasso_reg = Lasso()
        grid_search = GridSearchCV(lasso_reg, param_grid, cv=5)
        grid_result = grid_search.fit(self.val_x, self.val_y)
        # print(f"{self.pollution_type} Best: {grid_result.best_score_} using {grid_search.best_params_}")

    def knn_grid_cv(self):
        param_grid = {'n_neighbors': [10, 50, 100], 'weights': ('uniform', 'distance')}
        knn = neighbors.KNeighborsRegressor()
        grid_search = GridSearchCV(knn, param_grid, cv=5)
        grid_result = grid_search.fit(self.val_x, self.val_y)
        # print(f"{self.pollution_type} Best: {grid_result.best_score_} using {grid_search.best_params_}")

    def decision_tree_grid_cv(self):
        param_grid = {
            'criterion': ['poisson', 'absolute_error', 'friedman_mse', 'squared_error'],
            'max_depth': [30, 60, 100],
        }
        dt = DecisionTreeRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=dt, param_grid=param_grid,
                                   scoring='neg_mean_squared_error', cv=5, n_jobs=20, verbose=2)
        grid_result = grid_search.fit(self.val_x, self.val_y)
        # print(f"{self.pollution_type} Best: {grid_result.best_score_} using {grid_search.best_params_}")

    def random_forest_grid_cv(self):
        param_grid = {
            'n_estimators': [10, 100, 200, 500],
            'max_depth': [None, 5, 10, 20],
        }
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                                   scoring='neg_mean_squared_error', cv=5, n_jobs=20, verbose=2)
        grid_result = grid_search.fit(self.val_x, self.val_y)
        # print(f"{self.pollution_type} Best: {grid_result.best_score_} using {grid_search.best_params_}")

    def ada_boost_grid_cv(self):
        param_grid = {
            'n_estimators': [10, 50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.5],
            'loss': ['linear', 'square', 'exponential']
        }
        ada = AdaBoostRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=ada, param_grid=param_grid,
                                   scoring='neg_mean_squared_error', cv=5, n_jobs=20, verbose=2)
        grid_result = grid_search.fit(self.val_x, self.val_y)
        # print(f"{self.pollution_type} Best: {grid_result.best_score_} using {grid_search.best_params_}")

    def gradient_boosting_grid_cv(self):
        param_grid = {
            'n_estimators': [10, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 10],
            'max_features': [None, 'sqrt', 'log2']
        }

        gb = GradientBoostingRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=gb, param_grid=param_grid,
                                   scoring='neg_mean_squared_error', cv=5, n_jobs=20, verbose=2)
        grid_result = grid_search.fit(self.val_x, self.val_y)
        # print(f"{self.pollution_type} Best: {grid_result.best_score_} using {grid_search.best_params_}")

    def lightgbm_grid_cv(self):
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [10, 50, 100],
        }
        lgb_estimator = lgb.LGBMRegressor(boosting_type='gbdt', random_state=42)
        grid_search = GridSearchCV(estimator=lgb_estimator, param_grid=param_grid,
                                   scoring='neg_mean_squared_error', cv=5, n_jobs=20, verbose=2)
        grid_result = grid_search.fit(self.val_x, self.val_y)
        # print(f"{self.pollution_type} Best: {grid_result.best_score_} using {grid_search.best_params_}")

    def xgboost_grid_cv(self):
        param_grid = {
            'n_estimators': [10, 50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1]
        }
        xgb_estimator = XGBRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=xgb_estimator, param_grid=param_grid,
                                   scoring='neg_mean_squared_error', cv=5, n_jobs=20, verbose=2)
        grid_result = grid_search.fit(self.val_x, self.val_y)
        print(f"{self.pollution_type} Best: {grid_result.best_score_} using {grid_search.best_params_}")


def grid_cv_pollution(val_x, val_y, pollution_type:str):
    model_grid_search = ModelGridSearchCV(val_x, val_y, pollution_type)
    # model_grid_search.lasso_grid_cv()
    # model_grid_search.knn_grid_cv()
    # model_grid_search.decision_tree_grid_cv()
    # model_grid_search.random_forest_grid_cv()
    # model_grid_search.ada_boost_grid_cv()
    # model_grid_search.gradient_boosting_grid_cv()
    # model_grid_search.lightgbm_grid_cv()
    print('-------gridCV--------xgboost-----')
    model_grid_search.xgboost_grid_cv()

def train_stack(train_x, train_y, test_x, test_y, pollution_type: str, pollution_data_test,
                lasso_alpha, knn_n_neighbors, knn_weights,
                dt_max_depth, dt_criterion, rf_n_estimators, rf_max_depth,
                ada_n_estimators, ada_learning_rate,
                gbr_n_estimators, gbr_learning_rate, gbr_max_depth,
                lgb_n_estimators, lgb_learning_rate,
                xgb_n_estimators, xgb_learning_rate):

    # Train the models
    linear_reg = LinearRegression().fit(train_x, train_y)
    lasso_reg = Lasso(alpha=lasso_alpha).fit(train_x, train_y)
    knn_reg = neighbors.KNeighborsRegressor(n_neighbors=knn_n_neighbors,
                                            weights=knn_weights, n_jobs=20).fit(train_x, train_y)
    tree_reg = DecisionTreeRegressor(random_state=42, max_depth=dt_max_depth,
                                     criterion=dt_criterion).fit(train_x, train_y)
    rf_reg = RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=rf_max_depth,
                                   random_state=42, n_jobs=20).fit(train_x, train_y)
    ada_reg = AdaBoostRegressor(random_state=42, n_estimators=ada_n_estimators,
                                loss='exponential', learning_rate=ada_learning_rate).fit(train_x, train_y)
    gbr_reg = GradientBoostingRegressor(n_estimators=gbr_n_estimators, random_state=42,
                                        max_depth=gbr_max_depth, learning_rate=gbr_learning_rate,
                                        max_features='sqrt').fit(train_x, train_y)
    lgb_reg = lgb.LGBMRegressor(random_state=42, n_jobs=20, n_estimators=lgb_n_estimators,
                                learning_rate=lgb_learning_rate).fit(train_x, train_y)
    xgb_reg = XGBRegressor(random_state=42, n_jobs=20, n_estimators=xgb_n_estimators,
                           learning_rate=xgb_learning_rate).fit(train_x, train_y)

    # Predict on test set and stack predictions
    data2020_pred1 = predpnc(linear_reg, test_x, test_y, pollution_data_test)
    data2020_pred2 = predpnc(lasso_reg, test_x, test_y, data2020_pred1)
    data2020_pred4 = predpnc(knn_reg, test_x, test_y, data2020_pred2)
    data2020_pred5 = predpnc(tree_reg, test_x, test_y, data2020_pred4)
    data2020_pred6 = predpnc(rf_reg, test_x, test_y, data2020_pred5)
    data2020_pred7 = predpnc(ada_reg, test_x, test_y, data2020_pred6)
    data2020_pred8 = predpnc(gbr_reg, test_x, test_y, data2020_pred7)
    data2020_pred9 = predpnc(lgb_reg, test_x, test_y, data2020_pred8)
    data2020_pred10 = predpnc(xgb_reg, test_x, test_y, data2020_pred9)

    models = ['linear', 'lasso', 'knn', 'tree', 'rf', 'ada', 'gbr', 'lgb', 'xgb']
    new_columns = [f'{model}_{suffix}' for model in models for suffix in ['pred']]
    data2020_pred10.columns = list(pollution_data_test.columns) + new_columns
    data2020_pred10.to_csv(f'./out/pollution/model_stack/{pollution_type}_data_pred_test.csv')

    # Calculate metrics and save predictions on the training set
    train_results = pd.DataFrame(train_y, columns=['Actual'])
    for model, name in zip([linear_reg, lasso_reg, knn_reg, tree_reg,
                            rf_reg, ada_reg, gbr_reg, lgb_reg, xgb_reg], models):
        train_predictions = model.predict(train_x)
        train_results[f'{name}_pred'] = train_predictions
        train_r2 = r2_score(train_y, train_predictions)
        train_rmse = mean_squared_error(train_y, train_predictions, squared=False)
        train_mae = mean_absolute_error(train_y, train_predictions)
        print(f'{name} - Training R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}')
    
    train_results.to_csv(f'./out/pollution/model_train/{pollution_type}_train_results_test.csv', index=False)
    return linear_reg, lasso_reg, knn_reg, tree_reg, rf_reg, ada_reg, gbr_reg, lgb_reg, xgb_reg, data2020_pred10

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

def save_model(stack_reg, train_x, train_y, test_x, test_y, pollution_type: str, data_10):
    # Fit the stack model
    stack_fit = stack_reg.fit(train_x, train_y)
    
    # Predict on test set and save results
    data2020_pred11 = predpnc(stack_fit, test_x, test_y, data_10)
    new_columns = ['stack_pred']
    data2020_pred11.columns = list(data_10.columns) + new_columns
    data2020_pred11.to_csv(f'./out/pollution/model_stack/{pollution_type}_data_pred_stack.csv')
    
    # Predict on train set and save results
    train_predictions = stack_fit.predict(train_x)
    train_results = pd.DataFrame(train_y, columns=['Actual'])
    train_results['stack_pred'] = train_predictions
    train_results.to_csv(f'./out/pollution/model_train/{pollution_type}_train_results_stack.csv', index=False)
    
    # Calculate metrics on train set
    train_r2 = r2_score(train_y, train_predictions)
    train_rmse = mean_squared_error(train_y, train_predictions, squared=False)
    train_mae = mean_absolute_error(train_y, train_predictions)
    print(f'stack - Training R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}')
    
    # Save the model
    dump(stack_fit, f'./model/pollution/stack_trainedModel_{pollution_type}.joblib') 
    print('---------finish dump model---------------')

    return stack_fit, data2020_pred11

def train_and_save_stack_model(train_x, train_y, test_x, test_y,
                               estimators, final_estimator, pollution_type_short, data_10):

    stack_reg = StackingRegressor(estimators=estimators, final_estimator=final_estimator, cv=5)
    stack_pred = stack_reg.fit(train_x, train_y).predict(test_x)
    stack_mse, stack_mae, stack_r2, stack_evs = pred_plot(test_y, stack_pred)
    save_model(stack_reg, train_x, train_y, test_x, test_y, pollution_type_short, data_10)
    return stack_mse, stack_mae, stack_r2, stack_evs


def main(workstation_Flag: bool = False):
    pollution_types = ['PM2.5 [ug/m3]', 'PM10 [ug/m3]', 'NOX [ug/m3 eq. NO2]',
                    'NO2 [ug/m3]', 'O3 [ug/m3]']
    for pollution_type in pollution_types:
        (pollution_train_x, pollution_train_y, pollution_test_x, pollution_test_y,
        pollution_val_x, pollution_val_y) = scaler_pollution(pollution_type, workstation_Flag)
        grid_cv_pollution(pollution_val_x, pollution_val_y, pollution_type)
        _, pollution_data_test, _ = load_feature_data(workstation_Flag)
        params = storage_parameters(pollution_type)
        pollution_type_short = params.pop('short_name')
        (pollution_linear_reg, pollution_lasso_reg, pollution_knn_reg, pollution_tree_reg,
        pollution_rf_reg, pollution_ada_reg, pollution_gbr_reg,
        pollution_lgb_reg, pollution_xgb_reg, pollution_data_10) = train_stack(pollution_train_x,
                                                                pollution_train_y,
                                                                pollution_test_x,
                                                                pollution_test_y,
                                                                pollution_type_short,
                                                                pollution_data_test,
                                                                **params)
        if pollution_type_short == 'pm25':
            estimators = [("rf", pollution_rf_reg), ("lgb", pollution_lgb_reg),
                          ('xgb', pollution_xgb_reg)]
            final_estimator = MLPRegressor(hidden_layer_sizes=(10,), activation='relu',
                                           solver='adam', random_state=42, max_iter=100)
        elif pollution_type_short == 'pm10':
            estimators = [("lgb", pollution_lgb_reg), ('xgb', pollution_xgb_reg)]
            final_estimator = MLPRegressor(hidden_layer_sizes=(50, 50), activation='relu',
                                           solver='adam', random_state=42, max_iter=100)
        elif pollution_type_short == 'nox':
            estimators = [("lgb", pollution_lgb_reg), ('xgb', pollution_xgb_reg)]
            final_estimator = MLPRegressor(hidden_layer_sizes=(100,), activation='relu',
                                           solver='adam', random_state=42, max_iter=100)
        elif pollution_type_short == 'no2':
            estimators = [("lgb", pollution_lgb_reg), ('xgb', pollution_xgb_reg)]
            final_estimator = MLPRegressor(hidden_layer_sizes=(100,), activation='relu',
                                           solver='adam', random_state=42, max_iter=100)
        elif pollution_type_short == 'o3':
            estimators = [("lgb", pollution_lgb_reg), ('xgb', pollution_xgb_reg)]
            final_estimator = MLPRegressor(hidden_layer_sizes=(10,), activation='relu',
                                           solver='adam', random_state=42, max_iter=100)
        
        stack_mse, stack_mae, stack_r2, stack_evs = train_and_save_stack_model(
            pollution_train_x, pollution_train_y, pollution_test_x, pollution_test_y,
            estimators, final_estimator, pollution_type_short, pollution_data_10
        )
        print(f"{pollution_type} - MSE: {stack_mse}, MAE: {stack_mae}, R2: {stack_r2}, EVS: {stack_evs}")


if __name__ == '__main__':
    workstation_Flag = True
    main(workstation_Flag)