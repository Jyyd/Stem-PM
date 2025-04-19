'''
Author: JYYD jyyd23@mails.tsinghua.edu.cn
Date: 2024-09-12 02:20:27
LastEditors: JYYD jyyd23@mails.tsinghua.edu.cn
LastEditTime: 2024-09-13 02:07:44
FilePath: \pm_code\pollution_pop.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''
import os, sys
os.chdir(sys.path[0])
import rasterio
import numpy as np
import pandas as pd
import xarray as xr
from math import radians, cos, sin, asin, sqrt
from scipy.interpolate import griddata
import geopandas as gpd
from rasterio import features
from affine import Affine
from matplotlib.path import Path
from scipy.io import savemat, loadmat
from scipy import interpolate
from tqdm import trange
from datetime import datetime, timedelta
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor, as_completed


def get_swiss_pop_data(pop_filename):
    # load pop data
    popData = rasterio.open(pop_filename)
    pop = popData.read(1)
    popCoord = popData.transform
    cellLon = popCoord[0]
    cellLat = -popCoord[4]
    lonPop = np.arange(popCoord[2] + cellLon / 2, popCoord[0], cellLon/2)
    latPop = np.arange(popCoord[5] - cellLat / 2, popCoord[1], -cellLat/2)
    origin_lon = lonPop.reshape(-1, 1).T
    origin_lat = latPop.reshape(-1, 1).T
    origin_pop = pop
    # transform coordinates of pop
    def convert_coordinates(lon_array, lat_array):
        lon_array_converted = lon_array*2 + 180
        lat_array_converted = lat_array*2 - 90
        return lon_array_converted, lat_array_converted
    trans_origin_lon, trans_origin_lat = convert_coordinates(origin_lon, origin_lat)

    # select rectangular area around swiss
    lonfield = [5.7, 10.7]
    latfield = [45.6, 47.9]
    lon_indices = np.where((trans_origin_lon >= lonfield[0]) & (trans_origin_lon <= lonfield[1]))[1]
    lat_indices = np.where((trans_origin_lat >= latfield[0]) & (trans_origin_lat <= latfield[1]))[1]
    select_lon = trans_origin_lon[:, lon_indices]
    select_lat = trans_origin_lat[:, lat_indices]
    select_pop = origin_pop[np.ix_(lat_indices, lon_indices)] # use index select area
    
    # select the pop data in swiss
    switzerland = gpd.read_file('../code/pncEstimator-main/data/geoshp/gadm36_CHE_0.shp')
    switzerland_polygon = switzerland.unary_union
    paths = []
    if switzerland_polygon.geom_type == 'MultiPolygon':
        for polygon in switzerland_polygon.geoms:
            paths.append(Path(np.array(polygon.exterior.coords)))
    else:
        paths.append(Path(np.array(switzerland_polygon.exterior.coords)))

    select_lon_grid, select_lat_grid = np.meshgrid(select_lon, select_lat)
    select_lon_lat_points = np.vstack((select_lon_grid.flatten(), select_lat_grid.flatten())).T
    select_inside_swiss_mask = np.zeros(len(select_lon_lat_points), dtype=bool)
    for path in paths:
        select_inside_swiss_mask |= path.contains_points(select_lon_lat_points)
    select_inside_swiss_mask = select_inside_swiss_mask.reshape(select_pop.shape)
    inside_swiss_pop = np.where(select_inside_swiss_mask, select_pop, 0)
    select_lat = select_lat.flatten()
    select_lon = select_lon.flatten()
    # select_lat = select_lat[::-1] # reverse the lat
    
    return select_lon, select_lat, inside_swiss_pop

def cal_area(lat, lon):
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        # make sure the input is in radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * asin(sqrt(a))
        distance = R * c
        return distance
    area_matrix = np.zeros((len(lat), len(lon)))
    for i in range(len(lat) - 1):
        for j in range(len(lon) - 1):
            x_distance = haversine(lat[i], lon[j], lat[i], lon[j + 1])
            y_distance = haversine(lat[i], lon[j], lat[i + 1], lon[j])
            area_matrix[i, j] = x_distance * y_distance
    # print(area_matrix.shape)
    return area_matrix

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def calculate_overlap_area(pnc_lon, pnc_lat, pop_lon, pop_lat):
    pop_half_grid = 0.04167 / 2 # degree
    pnc_half_grid = 0.01 / 2 # degree

    left = max(pnc_lon - pnc_half_grid, pop_lon - pop_half_grid)
    right = min(pnc_lon + pnc_half_grid, pop_lon + pop_half_grid)
    bottom = max(pnc_lat - pnc_half_grid, pop_lat - pop_half_grid)
    top = min(pnc_lat + pnc_half_grid, pop_lat + pop_half_grid)

    overlap_width_lon = max(0, right - left)
    overlap_height_lat = max(0, top - bottom)

    return overlap_width_lon * overlap_height_lat # degree^2
    

def calculate_newpop(pnclon, pnclat, pnc_data, poplon_dens, poplat_dens, pop_data_dens, adjust_ratio):
    area_matrix_pnc = cal_area(pnclat, pnclon)
    pnc_area = area_matrix_pnc.mean() # km^2
    diff_pnc = 0.01*0.01 # degree^2
    newpop_values = np.zeros(pnc_data.shape)
    temp = []
    for i, pnc_lat in enumerate(pnclat):
        for j, pnc_lon in enumerate(pnclon):

            lon_idx, _ = find_nearest(poplon_dens, pnc_lon)
            lat_idx, _ = find_nearest(poplat_dens, pnc_lat)

            total_weighted_value = 0
            total_overlap_area = 0

            for di in [0, 1]:
                for dj in [0, 1]:
                    pop_i = lat_idx + di - 1
                    pop_j = lon_idx + dj - 1
                    if 0 <= pop_i < pop_data_dens.shape[0] and 0 <= pop_j < pop_data_dens.shape[1]:
                    # if 0 <= pop_i < len(poplat_dens) and 0 <= pop_j < len(poplon_dens):
                        overlap_area = calculate_overlap_area(pnc_lon, pnc_lat, poplon_dens[pop_j], poplat_dens[pop_i])
                        overlap_area = overlap_area * (pnc_area/diff_pnc)
                        
                        if overlap_area > 0:
                            weight = overlap_area
                            weighted_pop = pop_data_dens[pop_i, pop_j] * overlap_area * adjust_ratio
                            total_weighted_value += weighted_pop
                            total_overlap_area += overlap_area
            
            if total_overlap_area > 0.4:
                temp.append(total_overlap_area)
                newpop_values[i, j] = total_weighted_value / total_overlap_area * adjust_ratio
            else:
                newpop_values[i, j] = 0
    return newpop_values, total_overlap_area, temp

def load_mask_pop_data():
    # # density pop
    # total pop
    (
        poplon_dens_bt,
        poplat_dens_bt,
        pop_data_dens_bt
    ) = get_swiss_pop_data('../code/pop/tif_file/gpw_v4_basic_demographic_characteristics_rev11_atotpopbt_2010_dens_2pt5_min.tif')
    # male pop
    (
        poplon_dens_mt,
        poplat_dens_mt,
        pop_data_dens_mt
    ) = get_swiss_pop_data('../code/pop/tif_file/gpw_v4_basic_demographic_characteristics_rev11_atotpopmt_2010_dens_2pt5_min.tif')
    # famale pop
    (
        poplon_dens_ft,
        poplat_dens_ft,
        pop_data_dens_ft
    ) = get_swiss_pop_data('../code/pop/tif_file/gpw_v4_basic_demographic_characteristics_rev11_atotpopft_2010_dens_2pt5_min.tif')
    # age group pop
    # 0-14
    (
        poplon_dens_bt_0_14,
        poplat_dens_bt_0_14,
        pop_data_dens_bt_0_14) = get_swiss_pop_data('../code/pop/tif_file/gpw_v4_basic_demographic_characteristics_rev11_a000_014bt_2010_dens_2pt5_min.tif')
    # 15-64
    (
        poplon_dens_bt_15_64,
        poplat_dens_bt_15_64,
        pop_data_dens_bt_15_64
    ) = get_swiss_pop_data('../code/pop/tif_file/gpw_v4_basic_demographic_characteristics_rev11_a015_064bt_2010_dens_2pt5_min.tif')
    # 65+
    (
        poplon_dens_bt_65,
        poplat_dens_bt_65,
        pop_data_dens_bt_65
    ) = get_swiss_pop_data('../code/pop/tif_file/gpw_v4_basic_demographic_characteristics_rev11_a065plusbt_2010_dens_2pt5_min.tif')
    
    # # conut pop
    # total pop
    poplon_cntm_bt, poplat_cntm_bt, pop_data_cntm_bt = get_swiss_pop_data('../code/pop/tif_file/gpw_v4_basic_demographic_characteristics_rev11_atotpopbt_2010_cntm_2pt5_min.tif')
    # male pop
    poplon_cntm_mt, poplat_cntm_mt, pop_data_cntm_mt = get_swiss_pop_data('../code/pop/tif_file/gpw_v4_basic_demographic_characteristics_rev11_atotpopmt_2010_cntm_2pt5_min.tif')
    # famale pop
    poplon_cntm_ft, poplat_cntm_ft, pop_data_cntm_ft = get_swiss_pop_data('../code/pop/tif_file/gpw_v4_basic_demographic_characteristics_rev11_atotpopft_2010_cntm_2pt5_min.tif')
    # age group pop
    # 0-14
    poplon_cntm_bt_0_14, poplat_cntm_bt_0_14, pop_data_cntm_bt_0_14 = get_swiss_pop_data('../code/pop/tif_file/gpw_v4_basic_demographic_characteristics_rev11_a000_014bt_2010_cntm_2pt5_min.tif')
    # 15-64
    poplon_cntm_bt_15_64, poplat_cntm_bt_15_64, pop_data_cntm_bt_15_64 = get_swiss_pop_data('../code/pop/tif_file/gpw_v4_basic_demographic_characteristics_rev11_a015_064bt_2010_cntm_2pt5_min.tif')
    # 65+
    poplon_cntm_bt_65, poplat_cntm_bt_65, pop_data_cntm_bt_65 = get_swiss_pop_data('../code/pop/tif_file/gpw_v4_basic_demographic_characteristics_rev11_a065plusbt_2010_cntm_2pt5_min.tif')

    return (
        poplon_dens_bt, poplat_dens_bt, pop_data_dens_bt, 
        poplon_dens_mt, poplat_dens_mt, pop_data_dens_mt, 
        poplon_dens_ft, poplat_dens_ft, pop_data_dens_ft, 
        poplon_dens_bt_0_14, poplat_dens_bt_0_14, pop_data_dens_bt_0_14, 
        poplon_dens_bt_15_64, poplat_dens_bt_15_64, pop_data_dens_bt_15_64, 
        poplon_dens_bt_65, poplat_dens_bt_65, pop_data_dens_bt_65, 
        poplon_cntm_bt, poplat_cntm_bt, pop_data_cntm_bt, 
        poplon_cntm_mt, poplat_cntm_mt, pop_data_cntm_mt, 
        poplon_cntm_ft, poplat_cntm_ft, pop_data_cntm_ft, 
        poplon_cntm_bt_0_14, poplat_cntm_bt_0_14, pop_data_cntm_bt_0_14, 
        poplon_cntm_bt_15_64, poplat_cntm_bt_15_64, pop_data_cntm_bt_15_64, 
        poplon_cntm_bt_65, poplat_cntm_bt_65, pop_data_cntm_bt_65
    )

def print_pop_data_details(print_details:bool=False, pollution_type:str='PM2.5'):

    pnclon, pnclat = None, None
    (
        poplon_dens_bt, poplat_dens_bt, pop_data_dens_bt, 
        poplon_dens_mt, poplat_dens_mt, pop_data_dens_mt, 
        poplon_dens_ft, poplat_dens_ft, pop_data_dens_ft, 
        poplon_dens_bt_0_14, poplat_dens_bt_0_14, pop_data_dens_bt_0_14, 
        poplon_dens_bt_15_64, poplat_dens_bt_15_64, pop_data_dens_bt_15_64, 
        poplon_dens_bt_65, poplat_dens_bt_65, pop_data_dens_bt_65, 
        poplon_cntm_bt, poplat_cntm_bt, pop_data_cntm_bt, 
        poplon_cntm_mt, poplat_cntm_mt, pop_data_cntm_mt, 
        poplon_cntm_ft, poplat_cntm_ft, pop_data_cntm_ft, 
        poplon_cntm_bt_0_14, poplat_cntm_bt_0_14, pop_data_cntm_bt_0_14, 
        poplon_cntm_bt_15_64, poplat_cntm_bt_15_64, pop_data_cntm_bt_15_64, 
        poplon_cntm_bt_65, poplat_cntm_bt_65, pop_data_cntm_bt_65
    ) = load_mask_pop_data()

    pollution_mat_data = loadmat('../pm_code/out/mat_file/mean_pred_out/mean_pollutants_year.mat')
    pnclon = np.round(pollution_mat_data['lon'][0][:], 2)
    pnclat = pollution_mat_data['lat'][0][:]

    area_matrix_pnc = cal_area(pnclat, pnclon)
    pnc_area = area_matrix_pnc.mean() # km^2
    diff_pnc = 0.01*0.01 # degree^2
    pop_grid = 0.04167
    diff_pop = pop_grid*pop_grid # degree^2
    # bt
    area_matrix_pop_bt = cal_area(poplat_dens_bt, poplon_dens_bt)
    pop_area_bt = area_matrix_pop_bt.mean() # km^2
    adjust_ratio_bt = np.sum(pop_data_cntm_bt)/np.sum(pop_data_dens_bt*pop_area_bt)
    # mt
    area_matrix_pop_mt = cal_area(poplat_dens_mt, poplon_dens_mt)
    pop_area_mt = area_matrix_pop_mt.mean() # km^2
    adjust_ratio_mt = np.sum(pop_data_cntm_mt)/np.sum(pop_data_dens_mt*pop_area_mt)
    # ft
    area_matrix_pop_ft = cal_area(poplat_dens_ft, poplon_dens_ft)
    pop_area_ft = area_matrix_pop_ft.mean() # km^2
    adjust_ratio_ft = np.sum(pop_data_cntm_ft)/np.sum(pop_data_dens_ft*pop_area_ft)
    # bt 0-14
    area_matrix_pop_bt_0_14 = cal_area(poplat_dens_bt_0_14, poplon_dens_bt_0_14)
    pop_area_bt_0_14 = area_matrix_pop_bt_0_14.mean() # km^2
    adjust_ratio_bt_0_14 = np.sum(pop_data_cntm_bt_0_14)/np.sum(pop_data_dens_bt_0_14*pop_area_bt_0_14)
    # bt 15-64
    area_matrix_pop_bt_15_64 = cal_area(poplat_dens_bt_15_64, poplon_dens_bt_15_64)
    pop_area_bt_15_64 = area_matrix_pop_bt_15_64.mean() # km^2
    adjust_ratio_bt_15_64 = np.sum(pop_data_cntm_bt_15_64)/np.sum(pop_data_dens_bt_15_64*pop_area_bt_15_64)
    # bt 65+
    area_matrix_pop_bt_65 = cal_area(poplat_dens_bt_65, poplon_dens_bt_65)
    pop_area_bt_65 = area_matrix_pop_bt_65.mean() # km^2
    adjust_ratio_bt_65 = np.sum(pop_data_cntm_bt_65)/np.sum(pop_data_dens_bt_65*pop_area_bt_65)

    if print_details == True:
        print('pnc mean area : ', pnc_area)
        print('pnc km^2/degree^2 : ', pnc_area/diff_pnc)
        print('--------------------------------')
        # bt
        print('pop_area_unit_bt: ', pop_area_bt)
        print('pop cntm bt: ', np.sum(pop_data_cntm_bt)) # people
        print('pop_data_dens_bt*pop_area_bt: ', np.sum(pop_data_dens_bt*pop_area_bt))
        print('adjust_ratio_bt: ', adjust_ratio_bt)
        print('diff_bt ratio(%) : ', (np.sum(pop_data_dens_bt*pop_area_bt)-np.sum(pop_data_cntm_bt))/np.sum(pop_data_cntm_bt)*100)
        print('--------------------------------')
        # mt
        print('pop_area_unit_mt: ', pop_area_mt)
        print('pop cntm mt: ', np.sum(pop_data_cntm_mt)) # people
        print('pop_data_dens_mt*pop_area_mt: ', np.sum(pop_data_dens_mt*pop_area_mt))
        print('adjust_ratio_mt: ', adjust_ratio_mt)
        print('diff_mt ratio(%): ', (np.sum(pop_data_dens_mt*pop_area_mt)-np.sum(pop_data_cntm_mt))/np.sum(pop_data_cntm_mt)*100)
        print('--------------------------------')
        # ft
        print('pop_area_unit_ft: ', pop_area_ft)
        print('pop cntm ft: ', np.sum(pop_data_cntm_ft)) # people
        print('pop_data_dens_ft*pop_area_ft: ', np.sum(pop_data_dens_ft*pop_area_ft))
        print('adjust_ratio_ft: ', adjust_ratio_ft)
        print('diff_ft ratio(%): ', (np.sum(pop_data_dens_ft*pop_area_ft)-np.sum(pop_data_cntm_ft))/np.sum(pop_data_cntm_ft)*100)
        print('--------------------------------')
        # bt 0-14
        print('pop_area_unit_bt_0_14: ', pop_area_bt_0_14)
        print('pop cntm bt_0_14: ', np.sum(pop_data_cntm_bt_0_14)) # people
        print('pop_data_dens_bt_0_14*pop_area_bt_0_14: ', np.sum(pop_data_dens_bt_0_14*pop_area_bt_0_14))
        print('adjust_ratio_bt_0_14: ', adjust_ratio_bt_0_14)
        print('diff_bt_0_14 ratio(%): ', (np.sum(pop_data_dens_bt_0_14*pop_area_bt_0_14)-np.sum(pop_data_cntm_bt_0_14))/np.sum(pop_data_cntm_bt_0_14)*100)
        print('--------------------------------')
        # bt 15-64
        print('pop_area_unit_bt_15_64: ', pop_area_bt_15_64)
        print('pop cntm bt_15_64: ', np.sum(pop_data_cntm_bt_15_64)) # people
        print('pop_data_dens_bt_15_64*pop_area_bt_15_64: ', np.sum(pop_data_dens_bt_15_64*pop_area_bt_15_64))
        print('adjust_ratio_bt_15_64: ', adjust_ratio_bt_15_64)
        print('diff_bt_15_64 ratio(%): ', (np.sum(pop_data_dens_bt_15_64*pop_area_bt_15_64)-np.sum(pop_data_cntm_bt_15_64))/np.sum(pop_data_cntm_bt_15_64)*100)
        print('--------------------------------')
        # bt 65+
        print('pop_area_unit_bt_65: ', pop_area_bt_65)
        print('pop cntm bt_65: ', np.sum(pop_data_cntm_bt_65)) # people
        print('pop_data_dens_bt_65*pop_area_bt_65: ', np.sum(pop_data_dens_bt_65*pop_area_bt_65))
        print('adjust_ratio_bt_65: ', adjust_ratio_bt_65)
        print('diff_bt_65 ratio(%): ', (np.sum(pop_data_dens_bt_65*pop_area_bt_65)-np.sum(pop_data_cntm_bt_65))/np.sum(pop_data_cntm_bt_65)*100)
        print('--------------------------------')
    else:
        print('Details not printed. Set print_details=True to print details')

        return adjust_ratio_bt, adjust_ratio_mt, adjust_ratio_ft, adjust_ratio_bt_0_14, adjust_ratio_bt_15_64, adjust_ratio_bt_65


def get_pop_with_pnc(pollution_type:str='PM2.5'):

    pnclon, pnclat, pnc_data = None, None, None
    (
        poplon_dens_bt, poplat_dens_bt, pop_data_dens_bt, 
        poplon_dens_mt, poplat_dens_mt, pop_data_dens_mt, 
        poplon_dens_ft, poplat_dens_ft, pop_data_dens_ft, 
        poplon_dens_bt_0_14, poplat_dens_bt_0_14, pop_data_dens_bt_0_14, 
        poplon_dens_bt_15_64, poplat_dens_bt_15_64, pop_data_dens_bt_15_64, 
        poplon_dens_bt_65, poplat_dens_bt_65, pop_data_dens_bt_65, 
        poplon_cntm_bt, poplat_cntm_bt, pop_data_cntm_bt, 
        poplon_cntm_mt, poplat_cntm_mt, pop_data_cntm_mt, 
        poplon_cntm_ft, poplat_cntm_ft, pop_data_cntm_ft, 
        poplon_cntm_bt_0_14, poplat_cntm_bt_0_14, pop_data_cntm_bt_0_14, 
        poplon_cntm_bt_15_64, poplat_cntm_bt_15_64, pop_data_cntm_bt_15_64, 
        poplon_cntm_bt_65, poplat_cntm_bt_65, pop_data_cntm_bt_65
    ) = load_mask_pop_data()

    (
        adjust_ratio_bt, adjust_ratio_mt, adjust_ratio_ft,
        adjust_ratio_bt_0_14, adjust_ratio_bt_15_64, adjust_ratio_bt_65
    ) = print_pop_data_details(print_details=False, pollution_type=pollution_type)

    # pollution_mat_data = loadmat('../pm_code/out/mat_file/mean_pred_out/mean_pollutants_year.mat')
    # pnclon = np.round(pollution_mat_data['lon'][0][:], 2)
    # pnclat = np.round(pollution_mat_data['lat'][0][:], 2)
    # pm25_mat_data = pollution_mat_data['PM2.5'][0]
    # pnc_data = pm25_mat_data

    pollution_mat_data = loadmat('../pm_code/out/mat_file/mean_pred_out/mean_pollutants_2020_2023_month.mat')
    pnclon = np.round(pollution_mat_data['lon'][0][:], 2)
    pnclat = np.round(pollution_mat_data['lat'][0][:], 2)
    pm25_mat_data = pollution_mat_data['PM2.5'][0]
    pnc_data = pm25_mat_data

    newpop_values_bt, _,_ = calculate_newpop(pnclon, pnclat, pnc_data, poplon_dens_bt, poplat_dens_bt, pop_data_dens_bt, adjust_ratio_bt)
    print('pop total shape and sum : ', newpop_values_bt.shape, np.sum(newpop_values_bt))
    newpop_values_mt, _,_ = calculate_newpop(pnclon, pnclat, pnc_data, poplon_dens_mt, poplat_dens_mt, pop_data_dens_mt, adjust_ratio_mt)
    print('male pop shape and sum : ', newpop_values_mt.shape, np.sum(newpop_values_mt))
    newpop_values_ft, _,_ = calculate_newpop(pnclon, pnclat, pnc_data, poplon_dens_ft, poplat_dens_ft, pop_data_dens_ft, adjust_ratio_ft)
    print('female pop shape and sum : ', newpop_values_ft.shape, np.sum(newpop_values_ft))
    newpop_values_bt_0_14, _,_ = calculate_newpop(pnclon, pnclat, pnc_data, poplon_dens_bt_0_14, poplat_dens_bt_0_14, pop_data_dens_bt_0_14, adjust_ratio_bt_0_14)
    print('age 0-14 pop shape and sum : ', newpop_values_bt_0_14.shape, np.sum(newpop_values_bt_0_14))
    newpop_values_bt_15_64, _,_ = calculate_newpop(pnclon, pnclat, pnc_data, poplon_dens_bt_15_64, poplat_dens_bt_15_64, pop_data_dens_bt_15_64, adjust_ratio_bt_15_64)
    print('age 15-64 pop shape and sum : ', newpop_values_bt_15_64.shape, np.sum(newpop_values_bt_15_64))
    newpop_values_bt_65, _,_ = calculate_newpop(pnclon, pnclat, pnc_data, poplon_dens_bt_65, poplat_dens_bt_65, pop_data_dens_bt_65, adjust_ratio_bt_65)
    print('age 65+ pop shape and sum : ', newpop_values_bt_65.shape, np.sum(newpop_values_bt_65))

    data_save = np.stack((newpop_values_bt, newpop_values_mt,
                        newpop_values_ft, newpop_values_bt_0_14, newpop_values_bt_15_64,
                        newpop_values_bt_65), axis=0)
    
    savemat('../pm_code/out/mat_file/pop_mat/pop_2.mat',
            {'pop_bt': newpop_values_bt, 'pop_mt':  newpop_values_mt,
            'pop_ft': newpop_values_ft, 'pop_0_14': newpop_values_bt_0_14,
            'pop_15_64': newpop_values_bt_15_64, 'pop_65': newpop_values_bt_65})
    
    return data_save

def get_tif_data(tif_file):
    ds = xr.open_rasterio(tif_file)
    tiflon = np.array(ds['x'])
    tiflat = np.array(ds['y'])
    tifdata = np.array(ds.values[0])/255
    tifdata = np.ma.masked_where(tifdata == 1, tifdata)
    return tiflon, tiflat, tifdata

def save_tiff(save_tiff_name:str, save_tiff_data, lon_grid, lat_grid):
    transform = Affine.translation(lon_grid[0][0], lat_grid[0][0]) * Affine.scale(lon_grid[0,1]-lon_grid[0,0],
                                                                              lat_grid[1,0]-lat_grid[0,0])
    dataset = rasterio.open(
        save_tiff_name, 'w',
        driver='GTiff',
        height=lon_grid.shape[0],
        width=lat_grid.shape[1],
        count=1,
        dtype=save_tiff_data.dtype,
        crs='+proj=latlong',
        transform=transform,
    )
    dataset.write(save_tiff_data, 1)
    dataset.close()

def save_3d_tiff(save_tiff_name: str, save_tiff_data, lon_grid, lat_grid):
    transform = Affine.translation(lon_grid[0][0], lat_grid[0][0]) * Affine.scale(
        lon_grid[0, 1] - lon_grid[0, 0], lat_grid[1, 0] - lat_grid[0, 0]
    )
    
    with rasterio.open(
        save_tiff_name, 'w',
        driver='GTiff',
        height=lon_grid.shape[0],  # 行数（高度）
        width=lat_grid.shape[1],   # 列数（宽度）
        count=save_tiff_data.shape[0],  # 波段数量（即数组的第一个维度）
        dtype=save_tiff_data.dtype,
        crs='+proj=latlong',  # 使用WGS84坐标系
        transform=transform,
    ) as dataset:
        for i in range(save_tiff_data.shape[0]):
            # 写入每个波段的数据
            dataset.write(save_tiff_data[i, :, :], i + 1)
    
def get_district_data(tiff_filename:str, value_name:str, data):
    tiff_dataset = rasterio.open(tiff_filename)
    gdf = gpd.read_file('../code/pncEstimator-main/data/geoshp/gadm36_CHE_3.shp')
    
    weighted_data_by_district_mean = {}
    weighted_data_by_district_sum = {}
    weighted_data_by_district = {}
    for idx, row in gdf.iterrows():
        district_name = row['NAME_3']
        district_shape = row['geometry']

        mask = features.geometry_mask([district_shape], transform=tiff_dataset.transform, invert=True,
                                    out_shape=(tiff_dataset.height, tiff_dataset.width), all_touched=True)

        district_value = data[mask]
        weighted_data_by_district_sum[district_name] = np.sum(district_value)
        weighted_data_by_district_mean[district_name] = np.mean(district_value)
        weighted_data_by_district[district_name] = district_value
    
    weighted_district_mean = []
    weight_value_mean = []
    weight_value_sum = []
    for district, value in weighted_data_by_district_mean.items():
        weighted_district_mean.append(district)
        weight_value_mean.append(value)
    for district, value in weighted_data_by_district_sum.items():
        weight_value_sum.append(value)
    for district, value in weighted_data_by_district.items():
        weighted_data_by_district[district] = value
    
    tiff_dataset.close()

    weighted_data = pd.DataFrame(weighted_district_mean, columns=['District'])
    weighted_data[value_name + '_mean'] = weight_value_mean
    weighted_data[value_name + '_sum'] = weight_value_sum
    
    return weighted_data, weighted_data_by_district

def process_district(district_name, district_shape, data, transform):
    mask = features.geometry_mask([district_shape], transform=transform, invert=True,
                                  out_shape=(data.shape[1], data.shape[2]), all_touched=True)
    masked_data = data[:, mask]
    district_mean_values = np.mean(masked_data, axis=1)
    district_sum_values = np.sum(masked_data, axis=1)

    return (district_name, district_mean_values, district_sum_values)


def get_district_data_3d(tiff_filename: str, value_name: str, start_date: str, n_jobs=30):
    gdf = gpd.read_file('../code/pncEstimator-main/data/geoshp/gadm36_CHE_3.shp')

    with rasterio.open(tiff_filename) as tiff_dataset:
        data = tiff_dataset.read()
        transform = tiff_dataset.transform

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    dates = [start_date + timedelta(days=i) for i in range(data.shape[0])]
    date_columns_mean = [f"{value_name}_mean_{date.year}-{date.month:02d}-{date.day:02d}" for date in dates]
    date_columns_sum = [f"{value_name}_sum_{date.year}-{date.month:02d}-{date.day:02d}" for date in dates]

    results = Parallel(n_jobs=n_jobs)(delayed(process_district)(
        row['NAME_3'], row['geometry'], data, transform) for idx, row in gdf.iterrows())

    weighted_data_by_district_mean = {res[0]: res[1] for res in results}
    weighted_data_by_district_sum = {res[0]: res[2] for res in results}

    weighted_data_mean = pd.DataFrame.from_dict(weighted_data_by_district_mean, orient='index', columns=date_columns_mean)
    weighted_data_sum = pd.DataFrame.from_dict(weighted_data_by_district_sum, orient='index', columns=date_columns_sum)
    weighted_data = pd.concat([weighted_data_mean, weighted_data_sum], axis=1)

    return weighted_data

def save_annual_tiff():
    pollution_mat_data = loadmat('../pm_code/out/mat_file/mean_pred_out/mean_pollutants_year.mat')
    matlon = np.round(pollution_mat_data['lon'][0][:], 2)
    matlat = pollution_mat_data['lat'][0][:]
    lon_grid, lat_grid = np.meshgrid(matlon, matlat)
    pm25_mat_data = pollution_mat_data['PM2.5']
    pm10_mat_data = pollution_mat_data['PM10']
    nox_mat_data = pollution_mat_data['NOX']
    no2_mat_data = pollution_mat_data['NO2']
    o3_mat_data = pollution_mat_data['O3']

    pop_mat = loadmat('../pollution_out/pop.mat')
    pop_bt = pop_mat['pop_bt']
    pop_mt = pop_mat['pop_mt']
    pop_ft = pop_mat['pop_ft']
    pop_bt_0_14 = pop_mat['pop_0_14']
    pop_bt_15_64 = pop_mat['pop_15_64']
    pop_bt_65 = pop_mat['pop_65']

    weighted_data = []
    for i in range(11):
        print('Processing year : ', i+2013)
        pm25_mat_temp = np.nan_to_num(pm25_mat_data[i], nan=0.0)
        pm10_mat_temp = np.nan_to_num(pm10_mat_data[i], nan=0.0)
        nox_mat_temp = np.nan_to_num(nox_mat_data[i], nan=0.0)
        no2_mat_temp = np.nan_to_num(no2_mat_data[i], nan=0.0)
        o3_mat_temp = np.nan_to_num(o3_mat_data[i], nan=0.0)

        save_tiff('./out/tif_file/annual/pm25_'+str(i+2013)+'.tif', pm25_mat_temp, lon_grid, lat_grid)
        save_tiff('./out/tif_file/annual/pm10_'+str(i+2013)+'.tif', pm10_mat_temp, lon_grid, lat_grid)
        save_tiff('./out/tif_file/annual/nox_'+str(i+2013)+'.tif', nox_mat_temp, lon_grid, lat_grid)
        save_tiff('./out/tif_file/annual/no2_'+str(i+2013)+'.tif', no2_mat_temp, lon_grid, lat_grid)
        save_tiff('./out/tif_file/annual/o3_'+str(i+2013)+'.tif', o3_mat_temp, lon_grid, lat_grid)

        save_tiff('./out/tif_file/annual/pm25_pop_'+str(i+2013)+'.tif', pm25_mat_temp*pop_bt, lon_grid, lat_grid)
        save_tiff('./out/tif_file/annual/pm10_pop_'+str(i+2013)+'.tif', pm10_mat_temp*pop_bt, lon_grid, lat_grid)
        save_tiff('./out/tif_file/annual/nox_pop_'+str(i+2013)+'.tif', nox_mat_temp*pop_bt, lon_grid, lat_grid)
        save_tiff('./out/tif_file/annual/no2_pop_'+str(i+2013)+'.tif', no2_mat_temp*pop_bt, lon_grid, lat_grid)
        save_tiff('./out/tif_file/annual/o3_pop_'+str(i+2013)+'.tif', o3_mat_temp*pop_bt, lon_grid, lat_grid)

        weighted_pm25, weighted_pm25_district = get_district_data('./out/tif_file/annual/pm25_'+str(i+2013)+'.tif', f'pm25_{i+2013}', pm25_mat_temp)
        weighted_pm10, weighted_pm10_district = get_district_data('./out/tif_file/annual/pm10_'+str(i+2013)+'.tif', f'pm10_{i+2013}', pm10_mat_temp)
        weighted_nox, weighted_nox_district = get_district_data('./out/tif_file/annual/nox_'+str(i+2013)+'.tif', f'nox_{i+2013}', nox_mat_temp)
        weighted_no2, weighted_no2_district = get_district_data('./out/tif_file/annual/no2_'+str(i+2013)+'.tif', f'no2_{i+2013}', no2_mat_temp)
        weighted_o3, weighted_o3_district = get_district_data('./out/tif_file/annual/o3_'+str(i+2013)+'.tif', f'o3_{i+2013}', o3_mat_temp)

        weighted_pop_bt_pm25, _ = get_district_data('./out/tif_file/annual/pm25_pop_'+str(i+2013)+'.tif', f'pm25_pop_{i+2013}', pm25_mat_temp*pop_bt)
        weighted_pop_bt_pm10, _ = get_district_data('./out/tif_file/annual/pm10_pop_'+str(i+2013)+'.tif', f'pm10_pop_{i+2013}', pm10_mat_temp*pop_bt)
        weighted_pop_bt_nox, _ = get_district_data('./out/tif_file/annual/nox_pop_'+str(i+2013)+'.tif', f'nox_pop_{i+2013}', nox_mat_temp*pop_bt)
        weighted_pop_bt_no2, _ = get_district_data('./out/tif_file/annual/no2_pop_'+str(i+2013)+'.tif', f'no2_pop_{i+2013}', no2_mat_temp*pop_bt)
        weighted_pop_bt_o3, _ = get_district_data('./out/tif_file/annual/o3_pop_'+str(i+2013)+'.tif', f'o3_pop_{i+2013}', o3_mat_temp*pop_bt)

        if i == 10:
            save_tiff('./out/tif_file/pop/pop_bt.tif', pop_bt, lon_grid, lat_grid)
            save_tiff('./out/tif_file/pop/pop_mt.tif', pop_mt, lon_grid, lat_grid)
            save_tiff('./out/tif_file/pop/pop_ft.tif', pop_ft, lon_grid, lat_grid)
            save_tiff('./out/tif_file/pop/pop_bt_0_14.tif', pop_bt_0_14, lon_grid, lat_grid)
            save_tiff('./out/tif_file/pop/pop_bt_15_64.tif', pop_bt_15_64, lon_grid, lat_grid)
            save_tiff('./out/tif_file/pop/pop_bt_65.tif', pop_bt_65, lon_grid, lat_grid)
            weighted_pop_bt, weighted_pop_bt_district = get_district_data('./out/tif_file/pop/pop_bt.tif', 'pop_bt', pop_bt)
            weighted_pop_mt, weighted_pop_mt_district = get_district_data('./out/tif_file/pop/pop_mt.tif', 'pop_mt', pop_mt)
            weighted_pop_ft, weighted_pop_ft_district = get_district_data('./out/tif_file/pop/pop_ft.tif', 'pop_ft', pop_ft)
            weighted_pop_bt_0_14, weighted_pop_bt_0_14_district = get_district_data('./out/tif_file/pop/pop_bt_0_14.tif', 'pop_bt_0_14', pop_bt_0_14)
            weighted_pop_bt_15_64, weighted_pop_bt_15_64_district = get_district_data('./out/tif_file/pop/pop_bt_15_64.tif', 'pop_bt_15_64', pop_bt_15_64)
            weighted_pop_bt_65, weighted_pop_bt_65_district = get_district_data('./out/tif_file/pop/pop_bt_65.tif', 'pop_bt_65', pop_bt_65)
            dataframes = [weighted_pm25, weighted_pm10, weighted_nox,
                          weighted_no2, weighted_o3, 
                          weighted_pop_bt_pm25, weighted_pop_bt_pm10,
                          weighted_pop_bt_nox, weighted_pop_bt_no2, weighted_pop_bt_o3,
                          weighted_pop_bt, weighted_pop_mt, weighted_pop_ft,
                          weighted_pop_bt_0_14, weighted_pop_bt_15_64, weighted_pop_bt_65]
        else:
            dataframes = [weighted_pm25, weighted_pm10, weighted_nox,
                          weighted_no2, weighted_o3, 
                          weighted_pop_bt_pm25, weighted_pop_bt_pm10,
                          weighted_pop_bt_nox, weighted_pop_bt_no2, weighted_pop_bt_o3]
        
        weighted_data.append(dataframes)

    weighted_data_all = weighted_data[0][0]
    for dataframes in weighted_data:
        for df in dataframes[0:]:
            weighted_data_all = pd.merge(weighted_data_all, df, on='District', suffixes=('', '_drop'))
            weighted_data_all = weighted_data_all.loc[:, ~weighted_data_all.columns.str.endswith('_drop')]
    weighted_data_all.to_csv('./out/csv_file/test1/pop_weighted_data.csv', index=False)
    ajdusted_pop = np.sum(pop_bt)/np.sum(weighted_data_all['pop_bt_mean'])
    ajdusted_pop_mt = np.sum(pop_mt)/np.sum(weighted_data_all['pop_mt_mean'])
    ajdusted_pop_ft = np.sum(pop_ft)/np.sum(weighted_data_all['pop_ft_mean'])
    ajdusted_pop_bt_0_14 = np.sum(pop_bt_0_14)/np.sum(weighted_data_all['pop_bt_0_14_mean'])
    ajdusted_pop_bt_15_64 = np.sum(pop_bt_15_64)/np.sum(weighted_data_all['pop_bt_15_64_mean'])
    ajdusted_pop_bt_65 = np.sum(pop_bt_65)/np.sum(weighted_data_all['pop_bt_65_mean'])
    weighted_data_all['pop_bt_mean'] = weighted_data_all['pop_bt_mean']*ajdusted_pop
    weighted_data_all['pop_mt_mean'] = weighted_data_all['pop_mt_mean']*ajdusted_pop_mt
    weighted_data_all['pop_ft_mean'] = weighted_data_all['pop_ft_mean']*ajdusted_pop_ft
    weighted_data_all['pop_bt_0_14_mean'] = weighted_data_all['pop_bt_0_14_mean']*ajdusted_pop_bt_0_14
    weighted_data_all['pop_bt_15_64_mean'] = weighted_data_all['pop_bt_15_64_mean']*ajdusted_pop_bt_15_64
    weighted_data_all['pop_bt_65_mean'] = weighted_data_all['pop_bt_65_mean']*ajdusted_pop_bt_65
    weighted_data_all.to_csv('./out/csv_file/test1/pop_weighted_data.csv', index=False)
    return weighted_data_all

def crop_and_pad_array(array):
    cropped_array = array[5:-5, :]
    cropped_array = cropped_array[:, :-6]
    padded_array = np.pad(cropped_array, ((0, 0), (5, 0)), mode='constant', constant_values=0)
    return padded_array

def crop_and_pad_array_3d(array):
    if array.ndim != 3:
        raise ValueError("Input array must be 3-dimensional")
    cropped_array = array[:, 5:-5, :]
    cropped_array = cropped_array[:, :, :-6]
    padded_array = np.pad(cropped_array, ((0, 0), (0, 0), (5, 0)), mode='constant', constant_values=0)
    return padded_array

def save_annual_tiff_aqg():
    pollution_mat_data = loadmat('../pm_code/out/mat_file/mean_pred_out/mean_pollutants_year.mat')
    matlon = np.round(pollution_mat_data['lon'][0][:], 2)
    matlat = pollution_mat_data['lat'][0][:]
    lon_grid, lat_grid = np.meshgrid(matlon, matlat)
    # pm25_mat_data = pollution_mat_data['PM2.5']
    # pm10_mat_data = pollution_mat_data['PM10']
    # nox_mat_data = pollution_mat_data['NOX']
    # no2_mat_data = pollution_mat_data['NO2']
    # o3_mat_data = pollution_mat_data['O3']

    pop_mat = loadmat('../pm_code/out/mat_file/pop_mat/pop.mat')
    pop_bt = pop_mat['pop_bt']
    pop_mt = pop_mat['pop_mt']
    pop_ft = pop_mat['pop_ft']
    pop_bt_0_14 = pop_mat['pop_0_14']
    pop_bt_15_64 = pop_mat['pop_15_64']
    pop_bt_65 = pop_mat['pop_65']

    weighted_data = []
    weighted_data_aqg = []
    for i in range(11):
        print('Processing year : ', i+2013)
        # load pollutants annual data
        annual_data = loadmat(f'./out/mat_file/pred_out/daily_out/yearly_year/{i+2013}_annual_means.mat')
        pm25_annual_data = np.nan_to_num(annual_data['pm25_annual_mean'], nan=0.0)
        pm10_annual_data = np.nan_to_num(annual_data['pm10_annual_mean'], nan=0.0)
        nox_annual_data = np.nan_to_num(annual_data['nox_annual_mean'], nan=0.0)
        no2_annual_data = np.nan_to_num(annual_data['no2_annual_mean'], nan=0.0)
        o3_annual_data = np.nan_to_num(annual_data['o3_special_mean'], nan=0.0)

        # load daily data
        print('Loading daily data...')
        daily_data = loadmat(f'./out/mat_file/pred_out/daily_out/daily_year/{i+2013}_daily.mat')

        if i >=7:
            data_pm25_daily = crop_and_pad_array_3d(daily_data['pm25_daily'])
            data_pm10_daily = crop_and_pad_array_3d(daily_data['pm10_daily'])
            data_no2_daily = crop_and_pad_array_3d(daily_data['no2_daily'])
            data_nox_daily = crop_and_pad_array_3d(daily_data['nox_daily'])
            data_o3_mda8 = crop_and_pad_array_3d(daily_data['o3_daily'])
            pm25_annual_data  = crop_and_pad_array(pm25_annual_data)
            pm10_annual_data  = crop_and_pad_array(pm10_annual_data)
            nox_annual_data  = crop_and_pad_array(nox_annual_data)
            no2_annual_data  = crop_and_pad_array(no2_annual_data)
            o3_annual_data  = crop_and_pad_array(o3_annual_data)
        else:
            data_pm25_daily = daily_data['pm25_daily']
            data_pm10_daily = daily_data['pm10_daily']
            data_no2_daily = daily_data['no2_daily']
            data_nox_daily = daily_data['nox_daily']
            data_o3_mda8 = daily_data['o3_daily']

        # plus population
        print('Plus population...')
        data_pm25_daily_mult = data_pm25_daily * pop_bt[np.newaxis, :, :]
        data_pm10_daily_mult = data_pm10_daily * pop_bt[np.newaxis, :, :]
        data_no2_daily_mult = data_no2_daily * pop_bt[np.newaxis, :, :]
        data_nox_daily_mult = data_nox_daily * pop_bt[np.newaxis, :, :]
        data_o3_mda8_mult = data_o3_mda8 * pop_bt[np.newaxis, :, :]

        # 这里的数据是没有加上人口的
        # # load aqg data
        # daily_aqg_pm25_data = loadmat(f'./out/mat_file/pred_out/daily_out/daily/{i+2013}_PM2.5_daily.mat')
        # daily_aqg_pm10_data = loadmat(f'./out/mat_file/pred_out/daily_out/daily/{i+2013}_PM10_daily.mat')
        # daily_aqg_no2_data = loadmat(f'./out/mat_file/pred_out/daily_out/daily/{i+2013}_NO2_daily.mat')
        # daily_aqg_o3_data = loadmat(f'./out/mat_file/pred_out/daily_out/daily/{i+2013}_O3_dma8.mat')
        # # aqg level
        # daily_aqg_pm25 = daily_aqg_pm25_data['AQG']
        # daily_aqg_pm10 = daily_aqg_pm10_data['AQG']
        # daily_aqg_no2 = daily_aqg_no2_data['AQG']
        # daily_aqg_o3 = daily_aqg_o3_data['AQG']
        # # interim target
        # daily_it4_pm25 = daily_aqg_pm25_data['IT4']
        # daily_it4_pm10 = daily_aqg_pm10_data['IT4']
        # daily_it3_pm25 = daily_aqg_pm25_data['IT3']
        # daily_it3_pm10 = daily_aqg_pm10_data['IT3']
        # daily_it2_pm25 = daily_aqg_pm25_data['IT2']
        # daily_it2_pm10 = daily_aqg_pm10_data['IT2']
        # daily_it2_no2 = daily_aqg_no2_data['IT2']
        # daily_it2_o3 = daily_aqg_o3_data['IT2']
        # daily_it1_pm25 = daily_aqg_pm25_data['IT1']
        # daily_it1_pm10 = daily_aqg_pm10_data['IT1']
        # daily_it1_no2 = daily_aqg_no2_data['IT1']
        # daily_it1_o3 = daily_aqg_o3_data['IT1']
        
        print('Saving annual tiff files...')
        save_tiff(f'./tif_file/annual_o3spe/pm25_{i+2013}.tif', pm25_annual_data, lon_grid, lat_grid)
        save_tiff(f'./tif_file/annual_o3spe/pm10_{i+2013}.tif', pm10_annual_data, lon_grid, lat_grid)
        save_tiff(f'./tif_file/annual_o3spe/nox_{i+2013}.tif', nox_annual_data, lon_grid, lat_grid)
        save_tiff(f'./tif_file/annual_o3spe/no2_{i+2013}.tif', no2_annual_data, lon_grid, lat_grid)
        save_tiff(f'./tif_file/annual_o3spe/o3_{i+2013}.tif', o3_annual_data, lon_grid, lat_grid)

        save_tiff(f'./tif_file/annual_o3spe/pm25_pop_{i+2013}.tif', pm25_annual_data * pop_bt, lon_grid, lat_grid)
        save_tiff(f'./tif_file/annual_o3spe/pm10_pop_{i+2013}.tif', pm10_annual_data * pop_bt, lon_grid, lat_grid)
        save_tiff(f'./tif_file/annual_o3spe/nox_pop_{i+2013}.tif', nox_annual_data * pop_bt, lon_grid, lat_grid)
        save_tiff(f'./tif_file/annual_o3spe/no2_pop_{i+2013}.tif', no2_annual_data * pop_bt, lon_grid, lat_grid)
        save_tiff(f'./tif_file/annual_o3spe/o3_pop_{i+2013}.tif', o3_annual_data * pop_bt, lon_grid, lat_grid)

        print('Saving daily tiff files...')
        save_3d_tiff(f'./tif_file/annual_o3spe/pm25_daily_pop_{i+2013}.tif', data_pm25_daily_mult, lon_grid, lat_grid)
        save_3d_tiff(f'./tif_file/annual_o3spe/pm10_daily_pop_{i+2013}.tif', data_pm10_daily_mult, lon_grid, lat_grid)
        save_3d_tiff(f'./tif_file/annual_o3spe/nox_daily_pop_{i+2013}.tif', data_nox_daily_mult, lon_grid, lat_grid)
        save_3d_tiff(f'./tif_file/annual_o3spe/no2_daily_pop_{i+2013}.tif', data_no2_daily_mult, lon_grid, lat_grid)
        save_3d_tiff(f'./tif_file/annual_o3spe/o3_daily_pop_{i+2013}.tif', data_o3_mda8_mult, lon_grid, lat_grid)

        print('Processing annual data...')
        weighted_pm25, _ = get_district_data(f'./tif_file/annual_o3spe/pm25_{i+2013}.tif', f'pm25_{i+2013}', pm25_annual_data)
        weighted_pm10, _ = get_district_data(f'./tif_file/annual_o3spe/pm10_{i+2013}.tif', f'pm10_{i+2013}', pm10_annual_data)
        weighted_nox, _ = get_district_data(f'./tif_file/annual_o3spe/nox_{i+2013}.tif', f'nox_{i+2013}', nox_annual_data)
        weighted_no2, _ = get_district_data(f'./tif_file/annual_o3spe/no2_{i+2013}.tif', f'no2_{i+2013}', no2_annual_data)
        weighted_o3, _ = get_district_data(f'./tif_file/annual_o3spe/o3_{i+2013}.tif', f'o3_{i+2013}', o3_annual_data)

        weighted_pop_bt_pm25, _ = get_district_data(f'./tif_file/annual_o3spe/pm25_pop_{i+2013}.tif', f'pm25_pop_{i+2013}', pm25_annual_data * pop_bt)
        weighted_pop_bt_pm10, _= get_district_data(f'./tif_file/annual_o3spe/pm10_pop_{i+2013}.tif', f'pm10_pop_{i+2013}', pm10_annual_data * pop_bt)
        weighted_pop_bt_nox, _ = get_district_data(f'./tif_file/annual_o3spe/nox_pop_{i+2013}.tif', f'nox_pop_{i+2013}', nox_annual_data * pop_bt)
        weighted_pop_bt_no2, _ = get_district_data(f'./tif_file/annual_o3spe/no2_pop_{i+2013}.tif', f'no2_pop_{i+2013}', no2_annual_data * pop_bt)
        weighted_pop_bt_o3, _= get_district_data(f'./tif_file/annual_o3spe/o3_pop_{i+2013}.tif', f'o3_pop_{i+2013}', o3_annual_data * pop_bt)

        print('Processing daily data...')
        weighted_pm25_daily_pop = get_district_data_3d(f'./tif_file/annual_o3spe/pm25_daily_pop_{i+2013}.tif', f'day_pm25_pop_{i+2013}', f'{i+2013}-01-01')
        weighted_pm10_daily_pop = get_district_data_3d(f'./tif_file/annual_o3spe/pm10_daily_pop_{i+2013}.tif', f'day_pm10_pop_{i+2013}', f'{i+2013}-01-01')
        weighted_nox_daily_pop = get_district_data_3d(f'./tif_file/annual_o3spe/nox_daily_pop_{i+2013}.tif', f'day_nox_pop_{i+2013}', f'{i+2013}-01-01')
        weighted_no2_daily_pop = get_district_data_3d(f'./tif_file/annual_o3spe/no2_daily_pop_{i+2013}.tif', f'day_no2_pop_{i+2013}', f'{i+2013}-01-01')
        weighted_o3_daily_pop = get_district_data_3d(f'./tif_file/annual_o3spe/o3_daily_pop_{i+2013}.tif', f'day_o3_pop_{i+2013}', f'{i+2013}-01-01')
        
        if i == 10:
            # save_tiff('./out/tif_file/pop/pop_bt.tif', pop_bt, lon_grid, lat_grid)
            # save_tiff('./out/tif_file/pop/pop_mt.tif', pop_mt, lon_grid, lat_grid)
            # save_tiff('./out/tif_file/pop/pop_ft.tif', pop_ft, lon_grid, lat_grid)
            # save_tiff('./out/tif_file/pop/pop_bt_0_14.tif', pop_bt_0_14, lon_grid, lat_grid)
            # save_tiff('./out/tif_file/pop/pop_bt_15_64.tif', pop_bt_15_64, lon_grid, lat_grid)
            # save_tiff('./out/tif_file/pop/pop_bt_65.tif', pop_bt_65, lon_grid, lat_grid)
            weighted_pop_bt, weighted_pop_bt_district = get_district_data('./out/tif_file/pop/pop_bt.tif', 'pop_bt', pop_bt)
            weighted_pop_mt, weighted_pop_mt_district = get_district_data('./out/tif_file/pop/pop_mt.tif', 'pop_mt', pop_mt)
            weighted_pop_ft, weighted_pop_ft_district = get_district_data('./out/tif_file/pop/pop_ft.tif', 'pop_ft', pop_ft)
            weighted_pop_bt_0_14, weighted_pop_bt_0_14_district = get_district_data('./out/tif_file/pop/pop_bt_0_14.tif', 'pop_bt_0_14', pop_bt_0_14)
            weighted_pop_bt_15_64, weighted_pop_bt_15_64_district = get_district_data('./out/tif_file/pop/pop_bt_15_64.tif', 'pop_bt_15_64', pop_bt_15_64)
            weighted_pop_bt_65, weighted_pop_bt_65_district = get_district_data('./out/tif_file/pop/pop_bt_65.tif', 'pop_bt_65', pop_bt_65)
            dataframes = [weighted_pm25, weighted_pm10, weighted_nox,
                          weighted_no2, weighted_o3, 
                          weighted_pop_bt_pm25, weighted_pop_bt_pm10,
                          weighted_pop_bt_nox, weighted_pop_bt_no2, weighted_pop_bt_o3,
                          weighted_pm25_daily_pop, weighted_pm10_daily_pop,
                          weighted_nox_daily_pop, weighted_no2_daily_pop, weighted_o3_daily_pop,
                          weighted_pop_bt, weighted_pop_mt, weighted_pop_ft,
                          weighted_pop_bt_0_14, weighted_pop_bt_15_64, weighted_pop_bt_65]
            dataframes_aqg = [weighted_pm25_daily_pop, weighted_pm10_daily_pop,
                          weighted_nox_daily_pop, weighted_no2_daily_pop, weighted_o3_daily_pop]
        else:
            dataframes = [weighted_pm25, weighted_pm10, weighted_nox,
                          weighted_no2, weighted_o3, 
                          weighted_pop_bt_pm25, weighted_pop_bt_pm10,
                          weighted_pop_bt_nox, weighted_pop_bt_no2, weighted_pop_bt_o3,
                          weighted_pm25_daily_pop, weighted_pm10_daily_pop,
                          weighted_nox_daily_pop, weighted_no2_daily_pop, weighted_o3_daily_pop]
            dataframes_aqg = [weighted_pm25_daily_pop, weighted_pm10_daily_pop,
                          weighted_nox_daily_pop, weighted_no2_daily_pop, weighted_o3_daily_pop]
        
        weighted_data.append(dataframes)
        weighted_data_aqg.append(dataframes_aqg)

    weighted_data_all = weighted_data[0][0]
    for dataframes in weighted_data:
        for df in dataframes[0:]:
            weighted_data_all = pd.merge(weighted_data_all, df, on='District', suffixes=('', '_drop'))
            weighted_data_all = weighted_data_all.loc[:, ~weighted_data_all.columns.str.endswith('_drop')]
    weighted_data_all.to_csv('./out/csv_file/test2/pop_weighted_data_with_daily.csv', index=False)

    weighted_data_all_aqg = weighted_data_aqg[0][0]
    for dataframes in weighted_data_aqg:
        for df in dataframes[0:]:
            weighted_data_all_aqg = pd.merge(weighted_data_all_aqg, df, on='District', suffixes=('', '_drop'))
            weighted_data_all_aqg = weighted_data_all_aqg.loc[:, ~weighted_data_all_aqg.columns.str.endswith('_drop')]
    weighted_data_all_aqg.to_csv('./out/csv_file/test2/pop_weighted_data_aqg.csv', index=False)
    ajdusted_pop = np.sum(pop_bt)/np.sum(weighted_data_all['pop_bt_mean'])
    ajdusted_pop_mt = np.sum(pop_mt)/np.sum(weighted_data_all['pop_mt_mean'])
    ajdusted_pop_ft = np.sum(pop_ft)/np.sum(weighted_data_all['pop_ft_mean'])
    ajdusted_pop_bt_0_14 = np.sum(pop_bt_0_14)/np.sum(weighted_data_all['pop_bt_0_14_mean'])
    ajdusted_pop_bt_15_64 = np.sum(pop_bt_15_64)/np.sum(weighted_data_all['pop_bt_15_64_mean'])
    ajdusted_pop_bt_65 = np.sum(pop_bt_65)/np.sum(weighted_data_all['pop_bt_65_mean'])
    weighted_data_all['pop_bt_mean'] = weighted_data_all['pop_bt_mean']*ajdusted_pop
    weighted_data_all['pop_mt_mean'] = weighted_data_all['pop_mt_mean']*ajdusted_pop_mt
    weighted_data_all['pop_ft_mean'] = weighted_data_all['pop_ft_mean']*ajdusted_pop_ft
    weighted_data_all['pop_bt_0_14_mean'] = weighted_data_all['pop_bt_0_14_mean']*ajdusted_pop_bt_0_14
    weighted_data_all['pop_bt_15_64_mean'] = weighted_data_all['pop_bt_15_64_mean']*ajdusted_pop_bt_15_64
    weighted_data_all['pop_bt_65_mean'] = weighted_data_all['pop_bt_65_mean']*ajdusted_pop_bt_65
    weighted_data_all.to_csv('./out/csv_file/test2/pop_weighted_data_with_daily.csv', index=False)
    return weighted_data_all

def main():
    # # get pop mat data
    # get_pop_with_pnc()
    # save_annual_tiff()
    # run on the workstation
    save_annual_tiff_aqg()

if __name__ == '__main__':
    main()
