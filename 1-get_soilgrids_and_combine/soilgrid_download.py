import netCDF4 as nc
import multiprocessing as mp
from scipy import stats
import matplotlib.pyplot as plt
from soilgrids import SoilGrids
from tqdm import tqdm
import numpy as np
import time


# Calculate the coordinates to query soilgrids for

fn_p = 'forecast_data/agri4cast_precip/all_agri4cast_precip.nc'
fn_mx_t = 'forecast_data/agri4cast_max_temp/all_agri4cast_max_temp.nc'
fn_mn_t = 'forecast_data/agri4cast_min_temp/all_agri4cast_min_temp.nc'
fn_av_t = 'forecast_data/agri4cast_mean_temp/all_agri4cast_mean_temp.nc'

ds_p = nc.Dataset(fn_p)
ds_mx_t = nc.Dataset(fn_mx_t)
ds_mn_t = nc.Dataset(fn_mn_t)
ds_av_t = nc.Dataset(fn_av_t)

x = 0
lat = []
lon = []
for y in range(len(ds_mn_t['AirTemperatureMin'][x])):
    for v in range(len(ds_mn_t['AirTemperatureMin'][x][y])):
        val = ds_mn_t['AirTemperatureMin'][x][y][v]
        v2 = ds_mx_t['AirTemperatureMax'][x][y][v]
        v3 = ds_p['Rain'][x][y][v]
        v4 = ds_av_t['AirTemperatureMean'][x][y][v]
        if val is not np.ma.masked and v2 is not np.ma.masked and v3 is not np.ma.masked and v4 is not np.ma.masked :
            lat.append(ds_mn_t['lat'][y])
            lon.append(ds_mn_t['lon'][v])

# matrix lon, lat matrix of non masked values
jrc_XY = np.array([lon, lat]).T


def get_soilgrid_pt_data(soil_grids, coord, service_id, coverage_id, 
                 crs='urn:ogc:def:crs:EPSG::4326',
                 output='test.tif',
                 resolution=(100, 100),
                 square_len=0.31
                 ):
    # assuming the coord is the center of the grid point, we need to calc the max and min values from the center
    # coord = (lon, lat)
    displace = square_len/2
    
    n, s = coord[1] + displace, coord[1] - displace #  latitude  calcs
    e, w = coord[0] + displace, coord[0] - displace #  longitude calcs
    data = None
    while data is None:
        try:
            data = soil_grids.get_coverage_data(service_id=service_id, coverage_id=coverage_id, 
                                                west=float(w), south=float(s), east=float(e), north=float(n),  
                                                crs=crs,output=output, 
                                                height=resolution[0], width=resolution[1]
                                                )
        except:
            print(data)
            print('failed, attempting again')
            pass

    return data


def get_soilgrid_pt_mean(soil_grids, coord, service_id, coverage_id, fill=-32768,
                 crs='urn:ogc:def:crs:EPSG::4326',
                 output='test.tif',
                 resolution=(100, 100),
                 square_len=0.31,
                 ):
    # calculates the mean of a JRC grid point
    # steps:
    # 1. Calcualte bounds
    # 2. Query data
    # 3. Take the mean of the resulting data returned, returning None if there is no data
    # assuming the coord is the center of the grid point, we need to calc the max and min values from the center
    # coord = (lon, lat)
    displace = square_len/2
    
    n, s = coord[1] + displace, coord[1] - displace #  latitude calcs
    e, w = coord[0] + displace, coord[0] - displace #  latitude calcs
    data = None
    while data is None:
        try:
            data = soil_grids.get_coverage_data(service_id=service_id, coverage_id=coverage_id, 
                                                west=float(w), south=float(s), east=float(e), north=float(n),  
                                                crs=crs,output=output, 
                                                height=resolution[0], width=resolution[1]
                                                )
        except:
            print(data)
            print('failed, attempting again')
            pass

    mean_val = np.nanmean(data.values[data != fill].astype(np.float32))

    return mean_val

def get_soilgrid_pt_mode(soil_grids, coord, service_id, coverage_id, fill=255,
                 crs='urn:ogc:def:crs:EPSG::4326',
                 output='test.tif',
                 resolution=(100, 100),
                 square_len=0.31,
                 ):
    # steps:
    # 1. Calcualte bounds
    # 2. Query data
    # 3. Take the mode of the resulting data returned, returning None if there is no data
    # assuming the coord is the center of the grid point, we need to calc the max and min values from the center
    # coord = (lon, lat)
    displace = square_len/2
    
    n, s = coord[1] + displace, coord[1] - displace #  latitude calcs
    e, w = coord[0] + displace, coord[0] - displace #  latitude calcs
    data = None
    while data is None:
        try:
            data = soil_grids.get_coverage_data(service_id=service_id, coverage_id=coverage_id, 
                                                west=float(w), south=float(s), east=float(e), north=float(n),  
                                                crs=crs,output=output, 
                                                height=resolution[0], width=resolution[1]
                                                )
        except:
            print(data)
            print('failed, attempting again')
            pass

    filtered_data = data.values[data != fill]
    mode_val = stats.mode(filtered_data)

    return mode_val.mode


def get_soilgrid(soil_grids, coords, service_id, coverage_id, fill=-32768,
                 crs='urn:ogc:def:crs:EPSG::4326',
                 output='test.tif',
                 resolution=(100, 100),
                 square_len=0.31
                 ):
    # for each coordinate download its soildata
    # check which get function to use:
    get_function = get_soilgrid_pt_mode if service_id == 'wrb' else get_soilgrid_pt_mean
    data = []
    for coord in tqdm(coords, desc='Downloading Soilgrid'):
        r = None
        while r is None:
            try:
                r = get_function(soil_grids, coord, service_id, coverage_id, fill=fill,
                                crs=crs, output=output, 
                                resolution=resolution, square_len=square_len
                                )
            except:
                print('unexpected error, probably reading tif')
                pass
        data.append(r)

    return np.array(data)


def proc_soilgrid(proc_num, return_dict, coords, service_id, coverage_id, fill,
                 crs='urn:ogc:def:crs:EPSG::4326',
                 output='test.tif',
                 resolution=(100, 100),
                 square_len=0.31
                 ):
    # create soilgrid object, get the data from the coordinates
    soil_grids = SoilGrids()
    data = get_soilgrid(coords=coords, soil_grids=soil_grids, service_id=service_id, coverage_id=coverage_id, fill=fill, output=f'temp{proc_num}.tif')
    return_dict[proc_num] = data

def main():
    p_num = 16
    coverages = [
                'phh2o_0-5cm_mean', 'nitrogen_0-5cm_mean', 'sand_0-5cm_mean', 'clay_0-5cm_mean', 'silt_0-5cm_mean', 'cfvo_0-5cm_mean', 'soc_0-5cm_mean', 'ocs_0-30cm_mean', 'ocd_0-5cm_mean', 'bdod_0-5cm_mean', 'cec_0-5cm_mean', 
                'phh2o_5-15cm_mean', 'phh2o_15-30cm_mean', 'phh2o_30-60cm_mean', 'phh2o_60-100cm_mean', 'phh2o_100-200cm_mean', 
                'nitrogen_5-15cm_mean', 'nitrogen_15-30cm_mean', 'nitrogen_30-60cm_mean', 'nitrogen_60-100cm_mean', 'nitrogen_100-200cm_mean',
                'sand_5-15cm_mean', 'sand_15-30cm_mean', 'sand_30-60cm_mean', 'sand_60-100cm_mean', 'sand_100-200cm_mean', 
                'clay_5-15cm_mean', 'clay_15-30cm_mean', 'clay_30-60cm_mean', 'clay_60-100cm_mean', 'clay_100-200cm_mean', 
                'silt_5-15cm_mean', 'silt_15-30cm_mean', 'silt_30-60cm_mean', 'silt_60-100cm_mean', 'silt_100-200cm_mean', 
                'cfvo_5-15cm_mean', 'cfvo_15-30cm_mean', 'cfvo_30-60cm_mean', 'cfvo_60-100cm_mean', 'cfvo_100-200cm_mean', 
                'soc_5-15cm_mean', 'soc_15-30cm_mean', 'soc_30-60cm_mean', 'soc_60-100cm_mean', 'soc_100-200cm_mean', 
                'ocd_5-15cm_mean', 'ocd_15-30cm_mean', 'ocd_30-60cm_mean', 'ocd_60-100cm_mean', 'ocd_100-200cm_mean',
                'bdod_5-15cm_mean', 'bdod_15-30cm_mean', 'bdod_30-60cm_mean', 'bdod_60-100cm_mean', 'bdod_100-200cm_mean',
                'cec_5-15cm_mean', 'cec_15-30cm_mean', 'cec_30-60cm_mean', 'cec_60-100cm_mean', 'cec_100-200cm_mean',
                'MostProbable'
                ]
    services  = [
                'phh2o', 'nitrogen', 'sand', 'clay', 'silt', 'cfvo', 'soc', 'ocs', 'ocd', 'bdod', 'cec', 
                'phh2o','phh2o','phh2o','phh2o','phh2o',
                'nitrogen', 'nitrogen', 'nitrogen', 
                'nitrogen',
                'sand', 'sand', 'sand', 'sand', 'sand',
                'clay', 'clay', 'clay', 'clay', 'clay', 
                'silt', 'silt', 'silt', 'silt', 'silt', 
                'cfvo', 'cfvo', 'cfvo', 'cfvo', 'cfvo', 
                'soc', 'soc', 'soc', 'soc', 'soc', 
                'ocd', 'ocd', 'ocd', 'ocd', 'ocd',
                'bdod', 'bdod', 'bdod', 'bdod', 'bdod',
                'cec', 'cec', 'cec', 'cec', 'cec', 
                'wrb'
                ]
    split = np.array_split(jrc_XY, p_num)
    for service, coverage in zip(services, coverages):
        if service == 'wrb':
            fill = 255
            print(fill)
        else:
            fill = -32768
        manager = mp.Manager()
        return_dict = manager.dict()
        jobs = []
        for i in range(p_num):
            p = mp.Process(target=proc_soilgrid, args=(i, return_dict, split[i], service, coverage, fill))
            jobs.append(p)
            p.start()
        
        for proc in jobs:
            proc.join()
        
        vals = []
        for i in range(p_num):
            vals.append(return_dict[i])

        soil_data = np.concatenate(vals)
        np.save(file=f'soil_data/jrc_{coverage}', arr=soil_data)


if __name__ == "__main__":
    main()