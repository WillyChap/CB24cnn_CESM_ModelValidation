import argparse
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os 
import pandas as pd

def is_doyrange(doy,dd,tod,hh):
    daywind = 45
    if (dd - daywind) < 1:
        return ((doy >= (366+(dd-daywind))) | (doy <= dd+daywind)) & (tod==hh)
        
    elif (dd + daywind) > 366:
        return ((((doy <= 366) & (doy>=(dd-daywind))) | (doy <= (-1*((366-(dd-daywind))-(2*daywind)))  )))&(tod==hh)
    
    else:
        return ((doy >= dd-daywind) & (doy <= dd+daywind))&(tod==hh)
        
def is_leap_year(years):
    """Vectorized check for leap years, suitable for arrays."""
    return (years % 4 == 0) & ((years % 100 != 0) | (years % 400 == 0))
    
if __name__ == "__main__":
    # Setup the argument parser
    parser = argparse.ArgumentParser(description='Compute climatology from a specified NetCDF variable.')
    parser.add_argument('--dir_in', type=str, required=True, help='Input directory containing the NetCDF file.')
    parser.add_argument('--f_in', type=str, required=True, help='Filename of the NetCDF file.')
    parser.add_argument('--var_name', type=str, required=True, help='Name of the variable to process.')
    
    # Parse the arguments
    args = parser.parse_args()
    dir_in = args.dir_in
    f_in = args.f_in
    var_name = args.var_name
    
    file_path = os.path.join(dir_in, f_in)  # Join the directory and file name
    print('Im this file!!!!!: ', var_name, file_path)
    DS = xr.open_dataset(file_path)

    # Extract year from 'time' coordinate
    years = DS['time'].dt.year
    
    # Use the vectorized function to find leap years
    leap_year_mask = is_leap_year(years)
    
    # Apply mask to the years data and drop all non-leap years
    leap_years = years.where(leap_year_mask, drop=True)
    
    # Extract the first leap year
    if len(leap_years) ==0:
        first_leap_year = float(DS['time.year'][0])
    else:
        first_leap_year = leap_years.min().values  # Extract the minimum year, which is the first leap year
    
    print("First leap year in the dataset:", first_leap_year)

    DS_climo = xr.zeros_like(DS.sel(time=slice(f'{int(first_leap_year)}',f'{int(first_leap_year)}'))[var_name])
    print(f'... creating climo from centered 30 day average for {var_name}...')
    for ee,dayhr in (enumerate(DS.sel(time=slice(f'{int(first_leap_year)}',f'{int(first_leap_year+1)}')).time)):
        if ee%20 ==0:
            print('doing ',ee,' of 366')
        dooDOY = dayhr['time.dayofyear']
        hh=dayhr['time.hour']
        Dtemp = DS.sel(time=is_doyrange(DS['time.dayofyear'],dooDOY,DS['time.hour'],hh))[[var_name]].mean(['time'])
        
        DS_climo[ee,:,:] = Dtemp[var_name].values
    
        if ee == DS_climo.shape[0]-1:
            endee=ee
            enddate = dayhr
            break

    dir_out = f'{dir_in}/climo/'
    # Check if the directory exists
    if not os.path.exists(dir_out):
        # Create the directory
        os.makedirs(dir_out)
        print(f"Directory {dir_out} created.")
    
    DS_climo.to_dataset().to_netcdf(f'{dir_out}{var_name}_climo.nc')
    print('======= make anoms  =======')
    varsp=f_in.split(var_name)
    anom = DS[var_name].groupby('time.dayofyear')-DS_climo.groupby('time.dayofyear').mean("time")
    anom = anom.to_dataset()
    anom[var_name].attrs = DS[var_name].attrs
    anom.to_netcdf(f'{dir_out}{varsp[0]}{var_name}.anomalies{varsp[-1]}')

