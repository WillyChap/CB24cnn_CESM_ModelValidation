import argparse
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os 
import pandas as pd

def is_doyrange(doy,dd,tod,hh):
    daywind = 15
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
    parser.add_argument('--date_in', type=str, required=True, help='Name of the variable to process.')
    
    # Parse the arguments
    args = parser.parse_args()
    dir_in = args.dir_in
    f_in = args.f_in
    var_name = args.var_name
    date_in = pd.to_datetime(args.date_in)
    
    file_path = os.path.join(dir_in, f_in)  # Join the directory and file name
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
    dooDOY = date_in.dayofyear
    hh = date_in.hour
    DS_climo = DS.sel(time=is_doyrange(DS['time.dayofyear'],dooDOY,DS['time.hour'],hh))[[var_name]].mean(['time'])

    dir_out = f'{dir_in}/climo/'
    # Check if the directory exists
    if not os.path.exists(dir_out):
        # Create the directory
        os.makedirs(dir_out)
        print(f"Directory {dir_out} created.")
    
    DS_climo.to_netcdf(f'{dir_out}{var_name}_{date_in.strftime("%Y-%m-%d")}_climo.nc')
    

