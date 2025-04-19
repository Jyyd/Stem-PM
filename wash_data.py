import os, sys
os.chdir(sys.path[0])
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def get_NABEL_data(workstation_Flag: bool = False):
    # Define the data path based on the workstation flag
    if workstation_Flag:
        NABEL_raw_data = '../code/dataset/NABEL/raw_data_24/'
    else:
        NABEL_raw_data = '../code/pncEstimator-main/data/NABEL/raw_data_24/'
    
    # List all files in the directory
    filelist = os.listdir(NABEL_raw_data)
    print('The number of NABEL stations: ', len(filelist))
    NABELdata = []
    for file in filelist:
        # if file.startswith(('CHA', 'DAV')):
        #     continue
        filename = os.path.join(NABEL_raw_data, file)
        if filename.endswith('.csv'):
            data = pd.read_csv(filename, skiprows=6, sep=';')
            if 'PM2.5 [ug/m3]' not in data.columns:
                print(file)
                data['PM2.5 [ug/m3]'] = np.nan
            data_pm = data[['Date/time', 'O3 [ug/m3]', 'NO2 [ug/m3]', 'PM10 [ug/m3]',
                            'PM2.5 [ug/m3]', 'NOX [ug/m3 eq. NO2]', 'TEMP [C]', 'PREC [mm]',
                            'RAD [W/m2]']]
            station = [file[0:3]] * len(data_pm)
            data_pm['station'] = station
            # Reorder columns
            data_pm = data_pm[['Date/time', 'station', 'NOX [ug/m3 eq. NO2]', 'NO2 [ug/m3]',
                                'PM10 [ug/m3]', 'PM2.5 [ug/m3]',
                                'O3 [ug/m3]', 'TEMP [C]', 'PREC [mm]',
                                'RAD [W/m2]']]
            NABELdata.append(data_pm)
    NABELdata = pd.concat(NABELdata, axis=0).reset_index(drop=True)
    NABELdata['Date/time'] = pd.to_datetime(NABELdata['Date/time'], format='%d.%m.%Y %H:%M')
    # set the time zone
    NABELdata['NABEL_Time'] = NABELdata['Date/time'].dt.tz_localize('Europe/Zurich',
                                                                    ambiguous='NaT', nonexistent='shift_forward')
    NABELdata['UTC_Time'] = NABELdata['NABEL_Time'].dt.tz_convert('UTC')
    NABELdata['Date/time'] = NABELdata['UTC_Time'].dt.tz_localize(None)
    NABELdata = NABELdata.iloc[:, 0:10]
    print('++++++++++++++++++NABELdata++++++++++++++++++')
    print(NABELdata.head(2))
    print(NABELdata.columns)
    
    return NABELdata

def get_cams_data(workstation_Flag: bool = False):
    # cams data
    if workstation_Flag:
        camspath = '../code/dataset/CAMS/'
    else:
        camspath = '../code/pncEstimator-main/data/CAMS/'
    camsdata = pd.read_csv(camspath + 'camsdata_2.csv')
    camsdata.columns = ['stn', 'time_cams', 'pm10_cams', 'pm25_cams', 'o3_cams',
                        'no2_cams', 'no_cams', 'nox_cams']
    camsdata = camsdata[['stn', 'time_cams', 'pm10_cams', 'pm25_cams', 'o3_cams',
                        'no2_cams']]
    camsdata['time_cams'] = pd.to_datetime(camsdata['time_cams'], format='%Y-%m-%d %H:%M',errors='ignore')
    print('++++++++++++++++++camsdata++++++++++++++++++')
    print(camsdata.head(2))
    return camsdata

def get_meteo_data(workstation_Flag: bool = False):
    # meteo data
    if workstation_Flag:
        meteopath = '../code/dataset/meteo/meteo/'
    else:
        meteopath = '../code/pncEstimator-main/data/meteo/meteo/'
    meteodata = pd.read_csv(meteopath + 'meteodata16.txt', sep=';', dtype={1: str})
    meteodata.columns = ['stn', 'time', 'Radiation[W/m2] meteo', 'Temperature meteo', 'Precipitation[mm] meteo',
                        'Relative humidity[%] meteo', 'Wind speed[m/s] meteo', 'trafficVol']
    meteodata['time'] = pd.to_datetime(meteodata['time'], format='%Y-%m-%d %H:%M',errors='ignore')
    columns_to_convert = ['Radiation[W/m2] meteo', 'Temperature meteo',
                          'Precipitation[mm] meteo', 'Relative humidity[%] meteo', 'Wind speed[m/s] meteo']
    for column in columns_to_convert:
        meteodata[column] = pd.to_numeric(meteodata[column], errors='coerce')
    meteodata = meteodata.replace('-', np.nan)
    print('++++++++++++++++++meteodata++++++++++++++++++')
    print(meteodata.head(2))
    return meteodata

def get_merge_data(workstation_Flag: bool = False):
    NABELdata = get_NABEL_data(workstation_Flag)
    meteodata = get_meteo_data(workstation_Flag)
    camsdata = get_cams_data(workstation_Flag)
    # Ensure datetime columns are of the same type
    NABELdata['Date/time'] = pd.to_datetime(NABELdata['Date/time'], errors='coerce')
    camsdata['time_cams'] = pd.to_datetime(camsdata['time_cams'], errors='coerce')
    meteodata['time'] = pd.to_datetime(meteodata['time'], errors='coerce')
    # Remove rows with NaT in datetime columns
    NABELdata = NABELdata.drop_duplicates(subset=['station', 'Date/time'], keep='first')
    camsdata = camsdata.drop_duplicates(subset=['stn', 'time_cams'], keep='first')
    meteodata = meteodata.drop_duplicates(subset=['stn', 'time'], keep='first')
    # merge data
    pmdata = pd.merge(NABELdata, camsdata, how='left',
                  left_on=['station', 'Date/time'], right_on=['stn', 'time_cams'])
    pmdata = pd.merge(pmdata, meteodata, how='left',
                    left_on=['station', 'Date/time'], right_on=['stn', 'time'])
    duplicates = pmdata.duplicated(subset=['station', 'Date/time'], keep=False)
    print(f"Number of duplicate rows after merge: {duplicates.sum()}")
    pmdata = pmdata[['Date/time', 'station',
                    'PM2.5 [ug/m3]', 'pm25_cams',
                    'PM10 [ug/m3]', 'pm10_cams',
                    'NOX [ug/m3 eq. NO2]',
                    'NO2 [ug/m3]', 'no2_cams',
                    'O3 [ug/m3]', 'o3_cams',
                    'Radiation[W/m2] meteo', 'RAD [W/m2]',
                    'Temperature meteo', 'TEMP [C]',
                    'Precipitation[mm] meteo', 'PREC [mm]',
                    'Relative humidity[%] meteo', 'Wind speed[m/s] meteo',
                    'trafficVol']]
    pmdata['hour'] = pmdata['Date/time'].dt.hour
    pmdata['month'] = pmdata['Date/time'].dt.month
    pmdata['weekday'] = pmdata['Date/time'].dt.dayofweek
    pmdata['Radiation[W/m2] meteo'] = pd.to_numeric(pmdata['Radiation[W/m2] meteo'].fillna(pmdata['RAD [W/m2]']))
    pmdata['Temperature meteo'] = pd.to_numeric(pmdata['Temperature meteo'].fillna(pmdata['TEMP [C]']))
    pmdata['Precipitation[mm] meteo'] = pd.to_numeric(pmdata['Precipitation[mm] meteo'].fillna(pmdata['PREC [mm]']))
    pmdata = pmdata.reset_index(drop=True)
    return pmdata

def save_feature(workstation_Flag: bool = False):
    pmdata_washed = get_merge_data(workstation_Flag)
    pmdata = pmdata_washed.reset_index(drop=True)
    if workstation_Flag:
        save_feature_file = '../code/dataset/NABEL/feature_data/'
    else:
        save_feature_file = '../code/pncEstimator-main/data/NABEL/feature_eng/'
    pmdata.to_csv(save_feature_file + '/feature_data_PM_all_2.csv', index=False)


def main():
    workstation_Flag = False
    save_feature(workstation_Flag)
    print('Feature data saved successfully!')

if __name__ == "__main__":
    main()
