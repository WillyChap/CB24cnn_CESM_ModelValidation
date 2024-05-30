import argparse
import V_utils
import xarray as xr 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os 
import copy
from scipy import signal
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


import matplotlib.pyplot as plt
import matplotlib as mpl
#import collections
import matplotlib.ticker as mticker
from matplotlib import ticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import cartopy as cart
import cartopy.mpl.ticker as cticker
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cmocean
import scipy as sp
import warnings
warnings.filterwarnings("ignore")
import os



if __name__ == "__main__":
    SEAS='DJF'
    
    dir_exp = '/glade/derecho/scratch/wchapman/ADF/f.e.FTORCHmjo_CNNmjo/ts/'
    dir_cntrl = '/glade/derecho/scratch/wchapman/ADF/f.e.FTORCHmjo_CNTRLmjo/ts/'
    dir_obs = '/glade/campaign/cgd/amp/wchapman/Reanalysis/ERA5_obs/'
    
    fin_exp = 'f.e.FTORCHmjo_CNNmjo.cam.h0.Q.plev.197901-201012.nc'
    fin_cntrl = 'f.e.FTORCHmjo_CNTRLmjo.cam.h0.Q.plev.197901-201012.nc'
    fin_obs = 'ERA5.an.sfc.pl.camgrid.1979-2022.nc'
    
    var_cntrl = 'Q'
    var_exp = 'Q'
    var_obs = 'q'
    
    file_path_cntrl = os.path.join(dir_cntrl, fin_cntrl)  # Join the directory and file name
    file_path_exp = os.path.join(dir_exp, fin_exp)  # Join the directory and file name
    file_path_obs = os.path.join(dir_obs, fin_obs)  # Join the directory and file name
    
    DS_cntrl = xr.open_dataset(file_path_cntrl)
    DS_exp = xr.open_dataset(file_path_exp)
    DS_obs = xr.open_dataset(file_path_obs)
    
    #time issues: 
    datetimeindex = DS_cntrl.indexes['time'].to_datetimeindex()
    DS_cntrl['time'] = datetimeindex
    
    datetimeindex = DS_exp.indexes['time'].to_datetimeindex()
    DS_exp['time'] = datetimeindex
    
    DS_obs = DS_obs.rename({'level':'lev'}).sel(lev=DS_cntrl['lev'])
    
    #latitude weighting.
    weights_cos = np.cos(np.deg2rad(DS_cntrl.lat))
    weights_cos.name = "weights"
    
    print('...loading...')
    DS_obs = DS_obs[var_obs].to_dataset(name=var_obs).load()
    DS_cntrl = DS_cntrl[var_cntrl].to_dataset(name=var_cntrl).load()
    DS_exp = DS_exp[var_exp].to_dataset(name=var_exp).load()


    DS_exp =V_utils.fix_dates(DS_exp)
    DS_cntrl =V_utils.fix_dates(DS_cntrl)
    print('..dates fixed...')
    DS_cntrl_SA = V_utils.seasonal_averages(DS_cntrl, var_cntrl)
    print('..cntrl averaged...')
    DS_exp_SA = V_utils.seasonal_averages(DS_exp, var_exp)
    print('..exp averaged...')
    DS_obs_SA = V_utils.seasonal_averages(DS_obs, var_obs)
    print('..obs averaged...')


    #RMSE:
    RMSE_exp_SA = V_utils.rmse_field(DS_exp_SA[var_exp],DS_obs_SA[var_obs])
    RMSE_cntrl_SA = V_utils.rmse_field(DS_cntrl_SA[var_cntrl],DS_obs_SA[var_obs])


    # define the colormap
    cmap = plt.cm.RdYlBu_r
    # extract all colors from the Reds map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # make the first color entry to be whhite
    # make the last color entry to be most extreme--
    
    cmap.N
    # create the new map
    cmap = cmap.from_list('My cmap', cmaplist, cmap.N)
    clevels = np.arange(-1.1,1.1,.1)
    norm = mpl.colors.BoundaryNorm(clevels, cmap.N)


    for SEAS in ['DJF','MAM','JJA','SON']:
        plotter = (RMSE_cntrl_SA.sel(season=SEAS).mean('lon')-RMSE_exp_SA.sel(season=SEAS).mean('lon'))/(RMSE_cntrl_SA.sel(season=SEAS).mean('lon'))
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        lat = DS_cntrl_SA['lat']
        plevv = DS_cntrl_SA['lev']
        
        ff=plt.contourf(lat,plevv,plotter,levels=clevels,cmap=cmap,norm=norm,extend='both')
        
        plt.grid(True)
        plt.ylim([100,990])
        plt.xlabel('Latitude',fontsize=20)
        plt.ylabel('hPa',fontsize=20)
        lat_formatter = LatitudeFormatter(number_format='.0f')
        plt.tick_params(labelsize=18) 
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(lat_formatter)
        
        # create the colorbar
        ax2 = fig.add_axes([0.93, 0.1, 0.02, 0.8])
        cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, extend='both', spacing='proportional', ticks=clevels, boundaries=clevels)
        ax2.set_ylabel('[%] RMSE Reduction', size=25)
        cb.ax.tick_params(labelsize=20)
        tick_locator = ticker.MaxNLocator(nbins=10)
        cb.locator = tick_locator
        cb.update_ticks()
        
        
        ### +++ saving
        plt.savefig(f'{dir_exp}{SEAS}_{fin_exp}_vertical_RMSE.png', dpi=200,bbox_inches='tight')
        ### --- saving

    weighted_dict = {}
    for SEAS in ['DJF','MAM','JJA','SON']:
        weighted_rmse =[]
        for levs in RMSE_exp_SA['lev']:
            lv = int(levs.values)
        
            w1 = wgt_rmse(DS_cntrl_SA[var_exp].sel(lev=lv,season=SEAS),DS_obs_SA[var_obs].sel(lev=lv,season=SEAS),weights_cos)
            w2 = wgt_rmse(DS_exp_SA[var_exp].sel(lev=lv,season=SEAS),DS_obs_SA[var_obs].sel(lev=lv,season=SEAS),weights_cos)
    
            weighted_rmse.append(((w1-w2)/w1)*100)
    
        weighted_dict[SEAS]=weighted_rmse





    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot(weighted_dict['DJF'],np.array(RMSE_exp_SA['lev']),label='DJF',linewidth=4, alpha=0.9, color='navy')
    ax.plot(weighted_dict['SON'],np.array(RMSE_exp_SA['lev']),label='SON',linewidth=4, alpha=0.9, color='darkorange')
    ax.plot(weighted_dict['JJA'],np.array(RMSE_exp_SA['lev']),label='JJA',linewidth=4, alpha=0.9, color='crimson')
    ax.plot(weighted_dict['MAM'],np.array(RMSE_exp_SA['lev']),label='MAM',linewidth=4, alpha=0.9, color='teal')
    plt.ylim([200,1000])
    plt.xlim([-40,40])
    plt.xlabel('% RMSE Reduction',fontsize=20)
    plt.ylabel('hPa',fontsize=20)
    ax.invert_yaxis()
    plt.legend(fontsize=20)
    plt.tick_params(labelsize=18) 
    plt.grid(True)
    plt.show()
    ### +++ saving
    plt.savefig(f'{dir_exp}{SEAS}_{fin_exp}_AllSeason_vertical_RMSE.png', dpi=200,bbox_inches='tight')
    ### --- saving
    plt.show()
