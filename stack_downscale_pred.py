'''
Author: JYYD jyyd23@mails.tsinghua.edu.cn
Date: 2024-07-18 23:11:44
LastEditors: JYYD jyyd23@mails.tsinghua.edu.cn
LastEditTime: 2024-07-18 23:20:45
FilePath: \PNC\pm_code\stack_downscale_pred,py
Description: 

'''

import os, sys
os.chdir(sys.path[0])
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import xarray as xr
import scipy.io as sio
from tqdm import trange
from joblib import dump, load
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

def interpolate_data(cams_file, meteo_file, traffic_file, save_file, delta_x=0.01, delta_y=0.01, re_flag: bool=False):
    # Load CAMS data
    cams = xr.open_dataset(cams_file)
    cams_time = cams['time'].values
    if re_flag:
        cams_lon, cams_lat = cams['lon'].values, cams['lat'].values
        cams_pm10, cams_pm25, cams_o3, cams_no2 = cams['pm10'].values, cams['pm2p5'].values, cams['o3'].values, cams['no2'].values
    else:
        cams_lon, cams_lat = cams['longitude'].values, cams['latitude'].values # for
        cams_pm10, cams_pm25, cams_o3, cams_no2 = np.squeeze(cams['pm10_conc']).values, np.squeeze(cams['pm2p5_conc'].values), np.squeeze(cams['o3_conc'].values), np.squeeze(cams['no2_conc'].values)

    # Load meteo data
    meteo_data = xr.open_dataset(meteo_file)
    meteo_lon = meteo_data['longitude'].values
    meteo_lat = meteo_data['latitude'].values[::-1]
    meteo_time = meteo_data['time'].values
    radiation = np.maximum(np.diff(meteo_data['ssrd'].values, axis=0), 0) / 3600
    radiation = np.concatenate((np.zeros((1, *radiation.shape[1:])), radiation), axis=0)
    temperature = meteo_data['t2m'].values - 273.15
    precipitation = meteo_data['tp'].values * 1000
    precipitation = np.maximum(np.diff(precipitation, axis=0), 0)
    precipitation = np.concatenate((np.zeros((1, *precipitation.shape[1:])), precipitation), axis=0)
    dew = meteo_data['d2m'].values - 273.15
    humidity = 100 * np.exp(17.625 * dew / (243.04 + dew)) / np.exp(17.625 * temperature / (243.04 + temperature))
    u10, v10 = meteo_data['u10'].values, meteo_data['v10'].values
    speed = np.sqrt(u10**2 + v10**2)

    # Load traffic data
    traffic_data = sio.loadmat(traffic_file)
    lonRoad = traffic_data['lonRoad'].flatten()
    latRoad = traffic_data['latRoad'].flatten()
    trafficVol = traffic_data['trafficVol']

    # Determine the overlapping longitude and latitude ranges
    common_lon_min = max(cams_lon.min(), meteo_lon.min())
    common_lon_max = min(cams_lon.max(), meteo_lon.max())
    common_lat_min = max(cams_lat.min(), meteo_lat.min())
    common_lat_max = min(cams_lat.max(), meteo_lat.max())

    # Create new grid points based on the common ranges
    new_lon = np.arange(common_lon_min, common_lon_max, delta_x)
    new_lat = np.arange(common_lat_min, common_lat_max, delta_y)

    # Add additional latitude and longitude positions at the end
    additional_lat_end = new_lat[-1] + (new_lat[-1] - new_lat[-2])
    new_lat = np.append(new_lat, [additional_lat_end, additional_lat_end + (new_lat[-1] - new_lat[-2])])

    additional_lon_end = new_lon[-1] + (new_lon[-1] - new_lon[-2])
    new_lon = np.append(new_lon, [additional_lon_end, additional_lon_end + (new_lon[-1] - new_lon[-2])])

    new_lon_grid, new_lat_grid = np.meshgrid(new_lon, new_lat)

    # Initialize arrays to hold interpolated data
    new_pm10 = np.empty((cams_pm10.shape[0], new_lat_grid.shape[0], new_lon_grid.shape[1]))
    new_pm25 = np.empty((cams_pm25.shape[0], new_lat_grid.shape[0], new_lon_grid.shape[1]))
    new_o3 = np.empty((cams_o3.shape[0], new_lat_grid.shape[0], new_lon_grid.shape[1]))
    new_no2 = np.empty((cams_no2.shape[0], new_lat_grid.shape[0], new_lon_grid.shape[1]))
    new_radiation = np.empty((radiation.shape[0], new_lat_grid.shape[0], new_lon_grid.shape[1]))
    new_temperature = np.empty((temperature.shape[0], new_lat_grid.shape[0], new_lon_grid.shape[1]))
    new_precipitation = np.empty((precipitation.shape[0], new_lat_grid.shape[0], new_lon_grid.shape[1]))
    new_humidity = np.empty((humidity.shape[0], new_lat_grid.shape[0], new_lon_grid.shape[1]))
    new_speed = np.empty((speed.shape[0], new_lat_grid.shape[0], new_lon_grid.shape[1]))
    roadIntensity = np.empty((new_lat_grid.shape[0], new_lon_grid.shape[1]))

    # Ensure the loop range won't exceed the size of the data arrays
    num_time_steps = min(cams_pm10.shape[0], radiation.shape[0], temperature.shape[0], precipitation.shape[0], humidity.shape[0], speed.shape[0])

    # Function to interpolate data
    def interpolate(data, old_lon, old_lat, new_lon_grid, new_lat_grid):
        interpolator = RegularGridInterpolator((old_lat, old_lon), data, bounds_error=False, fill_value=np.nan)
        new_data = interpolator((new_lat_grid.flatten(), new_lon_grid.flatten())).reshape(new_lat_grid.shape)
        return np.nan_to_num(new_data, nan=0.0)

    # Interpolate CAMS and meteo data together
    for t in trange(num_time_steps):
        new_pm10[t] = interpolate(cams_pm10[t], cams_lon, cams_lat, new_lon_grid, new_lat_grid)
        new_pm25[t] = interpolate(cams_pm25[t], cams_lon, cams_lat, new_lon_grid, new_lat_grid)
        new_o3[t] = interpolate(cams_o3[t], cams_lon, cams_lat, new_lon_grid, new_lat_grid)
        new_no2[t] = interpolate(cams_no2[t], cams_lon, cams_lat, new_lon_grid, new_lat_grid)
        
        new_radiation[t] = interpolate(radiation[t], meteo_lon, meteo_lat, new_lon_grid, new_lat_grid)
        new_temperature[t] = interpolate(temperature[t], meteo_lon, meteo_lat, new_lon_grid, new_lat_grid)
        new_precipitation[t] = interpolate(precipitation[t], meteo_lon, meteo_lat, new_lon_grid, new_lat_grid)
        new_humidity[t] = interpolate(humidity[t], meteo_lon, meteo_lat, new_lon_grid, new_lat_grid)
        new_speed[t] = interpolate(speed[t], meteo_lon, meteo_lat, new_lon_grid, new_lat_grid)

    # Interpolate traffic data
    deltaRoad = lonRoad[1] - lonRoad[0]
    for i in range(new_lat_grid.shape[0]):
        for j in range(new_lon_grid.shape[1]):
            xtmp = new_lon_grid[i, j]
            ytmp = new_lat_grid[i, j]
            idl = round(max(((xtmp - delta_x / 2) - lonRoad[0] + deltaRoad / 2) / deltaRoad + 1, 1)) - 1
            idr = round(min(((xtmp + delta_x / 2) - lonRoad[0] + deltaRoad / 2) / deltaRoad + 1, len(lonRoad))) - 1
            idu = round(max((latRoad[0] + deltaRoad / 2 - (ytmp + delta_y / 2)) / deltaRoad + 1, 1)) - 1
            idd = round(min((latRoad[0] + deltaRoad / 2 - (ytmp - delta_y / 2)) / deltaRoad + 1, len(latRoad))) - 1
            vtmp = trafficVol[idu:idd + 1, idl:idr + 1]
            roadIntensity[i, j] = np.sum(vtmp) / vtmp.size if vtmp.size > 0 else 0

    # Save interpolated data to .mat file
    sio.savemat(save_file, {
        'time': cams_time[:num_time_steps],
        'pm10': new_pm10[:num_time_steps],
        'pm25': new_pm25[:num_time_steps],
        'o3': new_o3[:num_time_steps],
        'no2': new_no2[:num_time_steps],
        'radiation': new_radiation[:num_time_steps],
        'temperature': new_temperature[:num_time_steps],
        'precipitation': new_precipitation[:num_time_steps],
        'humidity': new_humidity[:num_time_steps],
        'speed': new_speed[:num_time_steps],
        'roadIntensity': roadIntensity,
        'lon': new_lon,
        'lat': new_lat
    })

def load_pnc_joblib():
    pm25_regr = load('./model/pollution/stack_trainedModel_pm25.joblib')
    pm10_regr = load('./model/pollution/stack_trainedModel_pm10.joblib')
    nox_regr = load('./model/pollution/stack_trainedModel_nox.joblib')
    no2_regr = load('./model/pollution/stack_trainedModel_no2.joblib')
    o3_regr = load('./model/pollution/stack_trainedModel_o3.joblib')
    return pm25_regr, pm10_regr, nox_regr, no2_regr, o3_regr

# load sclar
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

def scaler_pollution(pollution_type:str, pollution_data_train, pollution_data_test, pollution_data_val):
    scaler_std = StandardScaler()
    pollution_x_train, pollution_y_train = pollution_data_split(pollution_type, pollution_data_train)
    pollution_x_test, pollution_y_test = pollution_data_split(pollution_type, pollution_data_test)
    pollution_x_val, pollution_y_val = pollution_data_split(pollution_type, pollution_data_val)
    pollution_x_all = np.vstack((pollution_x_train, pollution_x_test, pollution_x_val))
    pollution_x_all_scaler = scaler_std.fit(pollution_x_all)

    return pollution_x_all_scaler

def load_pollution_data():
    # load the data
    pollution_data = pd.read_csv('../code/dataset/NABEL/feature_data/feature_data_PM_all.csv')
    pollution_data = pollution_data.dropna() # drop the rows with missing values
    pollution_data['Date/time'] = pd.to_datetime(pollution_data['Date/time']) # convert the date column to datetime

    pollution_data = pollution_data[(pollution_data['Date/time']>='2016-01-01 01:00')]
    pollution_data_val = pollution_data[(pollution_data['Date/time']>='2016-01-01 01:00') &
                                        (pollution_data['Date/time']<'2017-01-01 01:00')]
    pollution_data_train = pollution_data[(pollution_data['Date/time']>='2017-01-01 01:00') &
                                        (pollution_data['Date/time']<'2021-01-01 01:00')]
    pollution_data_train = pollution_data_train.reset_index(drop=True)
    pollution_data_test = pollution_data[(pollution_data['Date/time']>='2021-01-01 01:00') &
                                        (pollution_data['Date/time']<'2022-01-01 01:00')]
    pollution_data_test = pollution_data_test.reset_index(drop=True)

    pm25_regr, pm10_regr, nox_regr, no2_regr, o3_regr = load_pnc_joblib()
    scaler_x = scaler_pollution('PM10 [ug/m3]', pollution_data_train, pollution_data_test, pollution_data_val)

    return pm25_regr, pm10_regr, nox_regr, no2_regr, o3_regr, scaler_x

def pred_data(mat_file, pred_mat_file, scaler_x,
              pm25_regr, pm10_regr, nox_regr, no2_regr, o3_regr):
    data_load = sio.loadmat(mat_file)
    mat_lon = data_load['lon']
    mat_lat = data_load['lat']
    mat_time = data_load['time']
    pm25_data = data_load['pm25']
    pm10_data = data_load['pm10']
    no2_data = data_load['no2']
    o3_data = data_load['o3']
    radiation_data = data_load['radiation']
    temperature_data = data_load['temperature']
    precipitation_data = data_load['precipitation']
    humidity_data = data_load['humidity']
    speed_data = data_load['speed']
    trafficvol_data = data_load['roadIntensity']

    mat_time_flatten = pd.to_datetime(mat_time.flatten())
    hour_data = np.array(mat_time_flatten.hour)
    week_data = np.array(mat_time_flatten.weekday)
    month_data = np.array(mat_time_flatten.month)

    initial_shape_0, initial_shape_1, initial_shape_2 = pm25_data.shape

    pm25_data_flattened = pm25_data.reshape(-1, 1)
    pm10_data_flattened = pm10_data.reshape(-1, 1)
    no2_data_flattened = no2_data.reshape(-1, 1)
    o3_data_flattened = o3_data.reshape(-1, 1)
    radiation_data_flattened = radiation_data.reshape(-1, 1)
    temperature_data_flattened = temperature_data.reshape(-1, 1)
    precipitation_data_flattened = precipitation_data.reshape(-1, 1)
    humidity_data_flattened = humidity_data.reshape(-1, 1)
    speed_data_flattened = speed_data.reshape(-1, 1)

    repeated_traffic = np.tile(trafficvol_data, (initial_shape_0, 1, 1))
    flattened_repeated_traffic = repeated_traffic.reshape(-1, 1)

    expanded_hour_data = np.zeros((initial_shape_0, initial_shape_1, initial_shape_2))
    expanded_week_data = np.zeros((initial_shape_0, initial_shape_1, initial_shape_2))
    expanded_month_data = np.zeros((initial_shape_0, initial_shape_1, initial_shape_2))

    for i, value in enumerate(hour_data):
        expanded_hour_data[i, :, :] = value
    for i, value in enumerate(week_data):
        expanded_week_data[i, :, :] = value
    for i, value in enumerate(month_data):
        expanded_month_data[i, :, :] = value

    hour_data_flattened = expanded_hour_data.reshape(-1, 1)
    week_data_flattened = expanded_week_data.reshape(-1, 1)
    month_data_flattened = expanded_month_data.reshape(-1, 1)

    feature_data = np.concatenate([pm25_data_flattened, pm10_data_flattened, no2_data_flattened, o3_data_flattened,
                            radiation_data_flattened, temperature_data_flattened, precipitation_data_flattened,
                            humidity_data_flattened, speed_data_flattened, flattened_repeated_traffic,
                            hour_data_flattened, week_data_flattened, month_data_flattened], axis=1)

    scaler_feature = scaler_x.transform(feature_data)
    scaler_feature = np.nan_to_num(scaler_feature, nan=0.0)  # 在归一化之后应用nan_to_num

    pred_pm25_tmp = pm25_regr.predict(scaler_feature)
    pred_pm10_tmp = pm10_regr.predict(scaler_feature)
    pred_nox_tmp = nox_regr.predict(scaler_feature)
    pred_no2_tmp = no2_regr.predict(scaler_feature)
    pred_o3_tmp = o3_regr.predict(scaler_feature)

    pred_pm25 = pred_pm25_tmp.reshape(initial_shape_0, initial_shape_1, initial_shape_2)
    pred_pm10 = pred_pm10_tmp.reshape(initial_shape_0, initial_shape_1, initial_shape_2)
    pred_nox = pred_nox_tmp.reshape(initial_shape_0, initial_shape_1, initial_shape_2)
    pred_no2 = pred_no2_tmp.reshape(initial_shape_0, initial_shape_1, initial_shape_2)
    pred_o3 = pred_o3_tmp.reshape(initial_shape_0, initial_shape_1, initial_shape_2)

    print(pred_mat_file)

    data_to_save = {
        'lon': mat_lon,
        'lat': mat_lat,
        'time': mat_time,
        'pred_pm25': pred_pm25,
        'pred_pm10': pred_pm10,
        'pred_nox': pred_nox,
        'pred_no2': pred_no2,
        'pred_o3': pred_o3,
    }
    sio.savemat(pred_mat_file, data_to_save)

def get_train_data(re_flag):
    traffic_file = '../code/dataset/roadData/trafficVol.mat'
    meteo_path = '../code/dataset/meteo/ERA5/'
    if re_flag:
        cams_path = '../code/dataset/CAMS/CAMS_European_air_quality_reanalyses/re/'
    else:
        cams_path = '../code/dataset/CAMS/CAMS_European_air_quality_reanalyses/for/'
    cams_files = os.listdir(cams_path)

    for cams_file in cams_files:
        time_str = cams_file.split('.')[0]
        meteo_file = meteo_path + cams_file
        cams_file = cams_path + cams_file
        save_file = f'./pollution_out/mat_out/{time_str}.mat'
        interpolate_data(cams_file, meteo_file, traffic_file, save_file, re_flag)

def get_pred_data():
    pm25_regr, pm10_regr, nox_regr, no2_regr, o3_regr, scaler_x = load_pollution_data()
    mat_path = './pollution_out/mat_out/'
    mat_files = os.listdir(mat_path)
    for mat_file in mat_files:
        time_str = mat_file.split('.')[0]
        mat_file_full = mat_path + mat_file
        pred_mat_file = f'./pollution_out/pred_out/{time_str}.mat'
        pred_data(mat_file_full, pred_mat_file, scaler_x,
                pm25_regr, pm10_regr, nox_regr, no2_regr, o3_regr)
        
if __name__ == '__main__':
    re_flag = False
    get_train_data(re_flag)
    get_pred_data()