import argparse
import pickle
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm
from cartopy.mpl.ticker import LatitudeFormatter
import warnings
warnings.filterwarnings("ignore")
import V_utils  # Assuming this contains the necessary utility functions

def wgt_rmse(fld1, fld2, wgt):
    """Calculated the area-weighted RMSE.
    Inputs are 2-d spatial fields, fld1 and fld2 with the same shape.
    They can be xarray DataArray or numpy arrays.
    Input wgt is the weight vector, expected to be 1-d, matching length of one dimension of the data.
    Returns a single float value.
    """
    assert len(fld1.shape) == 2,     "Input fields must have exactly two dimensions."
    assert fld1.shape == fld2.shape, "Input fields must have the same array shape."
    # in case these fields are in dask arrays, compute them now.
    if hasattr(fld1, "compute"):
        fld1 = fld1.compute()
    if hasattr(fld2, "compute"):
        fld2 = fld2.compute()
    if isinstance(fld1, xr.DataArray) and isinstance(fld2, xr.DataArray):
        return (np.sqrt(((fld1 - fld2)**2).weighted(wgt).mean())).values.item()
    else:
        check = [len(wgt) == s for s in fld1.shape]
        if ~np.any(check):
            raise IOError(f"Sorry, weight array has shape {wgt.shape} which is not compatible with data of shape {fld1.shape}")
        check = [len(wgt) != s for s in fld1.shape]
        dimsize = fld1.shape[np.argwhere(check).item()]  # want to get the dimension length for the dim that does not match the size of wgt
        warray = np.tile(wgt, (dimsize, 1)).transpose()   # May need more logic to ensure shape is correct.
        warray = warray / np.sum(warray) # normalize
        wmse = np.nansum(warray * (fld1 - fld2)**2)
        return np.sqrt( wmse ).item()

def calculate_weights(DS):
    month_length = DS.time.dt.days_in_month
    weights = (month_length.groupby("time.season") / month_length.groupby("time.season").sum())
    np.testing.assert_allclose(weights.groupby("time.season").sum().values, np.ones(4))
    return weights

def seasonal_averages_(DS, weights):
    DS_weighted = (DS * weights).groupby("time.season").sum(dim="time", skipna=False)
    return DS_weighted

def parse_args():
    parser = argparse.ArgumentParser(description="Climate Data Analysis")
    parser.add_argument('--dir_exp', type=str, default='/glade/derecho/scratch/wchapman/ADF/f.e.FTORCHmjo_CNNmjo/ts/', help='Directory for experimental data')
    parser.add_argument('--dir_cntrl', type=str, default='/glade/derecho/scratch/wchapman/ADF/f.e.FTORCHmjo_CNTRLmjo/ts/', help='Directory for control data')
    parser.add_argument('--dir_obs', type=str, default='/glade/campaign/cgd/amp/wchapman/Reanalysis/ERA5_obs/', help='Directory for observational data')
    parser.add_argument('--fin_exp', type=str, default='f.e.FTORCHmjo_CNNmjo.cam.h0.Q.plev.197901-201012.nc', help='Filename for experimental data')
    parser.add_argument('--fin_cntrl', type=str, default='f.e.FTORCHmjo_CNTRLmjo.cam.h0.Q.plev.197901-201012.nc', help='Filename for control data')
    parser.add_argument('--fin_obs', type=str, default='ERA5.an.sfc.pl.camgrid.1979-2022.nc', help='Filename for observational data')
    parser.add_argument('--var_cntrl', type=str, default='Q', help='Variable name in control data')
    parser.add_argument('--var_exp', type=str, default='Q', help='Variable name in experimental data')
    parser.add_argument('--var_obs', type=str, default='q', help='Variable name in observational data')
    return parser.parse_args()


def load_datasets(args):
    print(f'loading {args.dir_cntrl}')
    file_path_cntrl = os.path.join(args.dir_cntrl, args.fin_cntrl)
    file_path_exp = os.path.join(args.dir_exp, args.fin_exp)
    file_path_obs = os.path.join(args.dir_obs, args.fin_obs)
    print('loading datasets')
    print(f'loading cntrl: {file_path_cntrl}')
    print(f'loading exp: {file_path_exp}')
    print(f'loading obs: {file_path_obs}')
    
    DS_cntrl = xr.open_dataset(file_path_cntrl).load()
    DS_exp = xr.open_dataset(file_path_exp).load()
    DS_obs = xr.open_dataset(file_path_obs).load()
    print('loaded')
    return DS_cntrl, DS_exp, DS_obs

def preprocess_data(DS_cntrl, DS_exp, DS_obs):
    DS_cntrl['time'] = DS_cntrl.indexes['time'].to_datetimeindex()
    DS_exp['time'] = DS_exp.indexes['time'].to_datetimeindex()
    DS_obs = DS_obs.rename({'level': 'lev'}).sel(lev=DS_cntrl['lev'])

    DS_exp = V_utils.fix_dates(DS_exp)
    DS_cntrl = V_utils.fix_dates(DS_cntrl)
    
    return DS_cntrl, DS_exp, DS_obs

def main():
    print('starting')
    args = parse_args()
    print(f'arguments: {args}')
    DS_cntrl, DS_exp, DS_obs = load_datasets(args)
    DS_cntrl, DS_exp, DS_obs = preprocess_data(DS_cntrl, DS_exp, DS_obs)

    DS_cntrl = DS_cntrl.load()
    DS_exp = DS_exp.load()
    DS_obs = DS_obs.load()

    #latitude weighting.
    weights_cos = np.cos(np.deg2rad(DS_cntrl.lat))
    weights_cos.name = "weights"

    bs_nummy = 100
    for bs in range(bs_nummy):
        fout = f'./Bootstrapped/{args.fin_exp}_AllSeason_MARITIME_vertical_RMSE_{bs:04}.pkl'
        
        if os.path.exists(fout):
            continue
        
        print(f'...doing...{bs}')
        # Assuming DS_cntrl, DS_exp, DS_obs are xarray Datasets with Dask enabled for parallel computation
        allyrs = np.arange(1979, 2011)
        subset = np.random.choice(allyrs, 27, replace=False)
        
        # Ensure data is subset correctly and efficiently
        DS_cntrl_bs = DS_cntrl.sel(time=DS_cntrl.time.dt.year.isin(subset))
        DS_exp_bs = DS_exp.sel(time=DS_exp.time.dt.year.isin(subset))
        DS_obs_bs = DS_obs.sel(time=DS_obs.time.dt.year.isin(subset))
        
        # Calculate weights once assuming all datasets have the same calendar
        weights = calculate_weights(DS_cntrl_bs)
        
        # Compute seasonal averages using precomputed weights
        DS_cntrl_SA_BS = seasonal_averages_(DS_cntrl_bs, weights)
        DS_exp_SA_BS = seasonal_averages_(DS_exp_bs, weights)
        DS_obs_SA_BS = seasonal_averages_(DS_obs_bs, weights)
        
        weighted_dict = {}
        for SEAS in ['DJF','MAM','JJA','SON']:
            weighted_rmse =[]
            w1_a = []
            w2_a = []
            for levs in DS_cntrl_SA_BS['lev']:
                lv = int(levs.values)

                cntrlin = DS_cntrl_SA_BS[args.var_exp].sel(lev=lv,season=SEAS,lat=slice(-20,20),lon=slice(60,220))
                expin = DS_exp_SA_BS[args.var_exp].sel(lev=lv,season=SEAS,lat=slice(-20,20),lon=slice(60,220))
                obsin = DS_obs_SA_BS[args.var_obs].sel(lev=lv,season=SEAS,lat=slice(-20,20),lon=slice(60,220))
                win = weights_cos.sel(lat=slice(-20,20))
            
                w1 = wgt_rmse(cntrlin,obsin,win)
                w2 = wgt_rmse(expin,obsin,win)
        
                weighted_rmse.append(((w1-w2)/w1)*100)
                w1_a.append(w1)
                w2_a.append(w2)
        
            weighted_dict[SEAS]=weighted_rmse
            weighted_dict[f'{SEAS}_cntrl'] = w1_a
            weighted_dict[f'{SEAS}_exp'] = w2_a
            
        # Pickle the dictionary
        with open(fout, 'wb') as file:
            pickle.dump(weighted_dict, file)

if __name__ == "__main__":
    main()
