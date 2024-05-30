import argparse
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
    file_path_cntrl = os.path.join(args.dir_cntrl, args.fin_cntrl)
    file_path_exp = os.path.join(args.dir_exp, args.fin_exp)
    file_path_obs = os.path.join(args.dir_obs, args.fin_obs)
    print('loading datasets')
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
    DS_cntrl, DS_exp, DS_obs = load_datasets(args)
    DS_cntrl, DS_exp, DS_obs = preprocess_data(DS_cntrl, DS_exp, DS_obs)

    #latitude weighting.
    weights_cos = np.cos(np.deg2rad(DS_cntrl.lat))
    weights_cos.name = "weights"

    # Use the specific variables from the arguments
    print('seasonal averaging')
    DS_cntrl_SA = V_utils.seasonal_averages(DS_cntrl, args.var_cntrl)
    print('....1 of 3...')
    DS_exp_SA = V_utils.seasonal_averages(DS_exp, args.var_exp)
    print('....2 of 3...')
    DS_obs_SA = V_utils.seasonal_averages(DS_obs, args.var_obs)
    print('.... done ...')
    print('RMSE')
    RMSE_exp_SA = V_utils.rmse_field(DS_exp_SA[args.var_exp], DS_obs_SA[args.var_obs])
    print('RMSE')
    RMSE_cntrl_SA = V_utils.rmse_field(DS_cntrl_SA[args.var_cntrl], DS_obs_SA[args.var_obs])
    

    # Additional plotting or data handling logic goes here


    # define the colormap
    cmap = plt.cm.RdYlBu_r
    # extract all colors from the Reds map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # make the first color entry to be whhite
    # make the last color entry to be most extreme--
    
    cmap.N
    # create the new map
    cmap = cmap.from_list('My cmap', cmaplist, cmap.N)
    clevels = np.arange(-100,105,5)
    norm = mpl.colors.BoundaryNorm(clevels, cmap.N)


    for SEAS in ['DJF','MAM','JJA','SON']:
        plotter = 100*((RMSE_cntrl_SA.sel(season=SEAS).mean('lon')-RMSE_exp_SA.sel(season=SEAS).mean('lon'))/(RMSE_cntrl_SA.sel(season=SEAS).mean('lon')))
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        lat = DS_cntrl_SA['lat']
        plevv = DS_cntrl_SA['lev']
        
        ff=plt.contourf(lat,plevv,plotter,levels=clevels,cmap=cmap,norm=norm,extend='both')
        
        plt.grid(True)
        plt.ylim([10,1000])
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
        plt.savefig(f'{args.dir_exp}{SEAS}_{args.fin_exp}_vertical_RMSE.png', dpi=200,bbox_inches='tight')
        ### --- saving
        plt.show()

    weighted_dict = {}
    for SEAS in ['DJF','MAM','JJA','SON']:
        weighted_rmse =[]
        for levs in RMSE_exp_SA['lev']:
            lv = int(levs.values)
        
            w1 = V_utils.wgt_rmse(DS_cntrl_SA[args.var_exp].sel(lev=lv,season=SEAS),DS_obs_SA[args.var_obs].sel(lev=lv,season=SEAS),weights_cos)
            w2 = V_utils.wgt_rmse(DS_exp_SA[args.var_exp].sel(lev=lv,season=SEAS),DS_obs_SA[args.var_obs].sel(lev=lv,season=SEAS),weights_cos)
    
            weighted_rmse.append(((w1-w2)/w1)*100)
    
        weighted_dict[SEAS]=weighted_rmse


    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot(weighted_dict['DJF'],np.array(RMSE_exp_SA['lev']),label='DJF',linewidth=4, alpha=0.9, color='navy')
    ax.plot(weighted_dict['SON'],np.array(RMSE_exp_SA['lev']),label='SON',linewidth=4, alpha=0.9, color='darkorange')
    ax.plot(weighted_dict['JJA'],np.array(RMSE_exp_SA['lev']),label='JJA',linewidth=4, alpha=0.9, color='crimson')
    ax.plot(weighted_dict['MAM'],np.array(RMSE_exp_SA['lev']),label='MAM',linewidth=4, alpha=0.9, color='teal')
    plt.ylim([10,1000])
    plt.xlim([-75,75])
    plt.xlabel('% RMSE Reduction',fontsize=20)
    plt.ylabel('hPa',fontsize=20)
    ax.invert_yaxis()
    plt.legend(fontsize=20)
    plt.tick_params(labelsize=18) 
    plt.grid(True)
    ### +++ saving
    plt.savefig(f'{args.dir_exp}{SEAS}_{args.fin_exp}_AllSeason_vertical_RMSE.png', dpi=200,bbox_inches='tight')
    ### --- saving
    plt.show()



if __name__ == "__main__":
    main()
