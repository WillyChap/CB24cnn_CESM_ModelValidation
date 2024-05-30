import xarray as xr 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os 
import scipy as sp
from scipy.signal import butter, lfilter
from scipy.signal import convolve
import xesmf as xe
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


def bandpass_filter_via_fft_1d(time_series, low_period, high_period, sampling_interval=1):
    """
    Apply a bandpass filter to a time series using FFT and inverse FFT.
    
    Parameters:
    - time_series: numpy array, the input time series data.
    - low_period: float, the lower bound of the period (in days) to allow through the filter.
    - high_period: float, the upper bound of the period (in days) to allow through the filter.
    - sampling_interval: float, the sampling interval of the time series (default is 1 day).
    
    Returns:
    - filtered_time_series: numpy array, the bandpass-filtered time series.
    """
    # Perform FFT on the original data
    fft_data = np.fft.fft(time_series)
    n = len(fft_data)
    
    # Generate frequency axis and define bandpass frequency limits
    frequencies = np.fft.fftfreq(n, d=sampling_interval)
    f_low = 1 / high_period  # Convert period to frequency
    f_high = 1 / low_period  # Convert period to frequency

    # Create a bandpass filter mask
    bandpass_mask = (np.abs(frequencies) >= f_low) & (np.abs(frequencies) <= f_high)

    # Apply the mask to the FFT data
    filtered_fft_data = fft_data * bandpass_mask

    # Apply inverse FFT to get the filtered time series back in the time domain
    filtered_time_series = np.fft.ifft(filtered_fft_data).real

    return filtered_time_series

# Example usage:
# Assuming 'sickos' is a numpy array representing your time series data
# low_period = 100 days, high_period = 20 days

def bandpass_filter_via_fft_2d(data, low_period, high_period, sampling_interval=1):
    """
    Apply a bandpass filter to a 3D time series (time, lat, lon) using FFT and inverse FFT.
    
    Parameters:
    - data: numpy array, the input time series data of shape (time, lat, lon).
    - low_period: float, the lower bound of the period (in days) to allow through the filter.
    - high_period: float, the upper bound of the period (in days) to allow through the filter.
    - sampling_interval: float, the sampling interval of the time series (default is 1 day).
    
    Returns:
    - filtered_data: numpy array, the bandpass-filtered time series.
    """
    # Initialize the output array with the same shape as input data
    filtered_data = np.empty_like(data)

    # Loop over each spatial point
    for i in range(data.shape[1]):  # latitude
        # Extract the time series at each point
        time_series = data[:, i]
            
        # Perform FFT on the original data
        fft_data = np.fft.fft(time_series)
        n = len(fft_data)
            
        # Generate frequency axis and define bandpass frequency limits
        frequencies = np.fft.fftfreq(n, d=sampling_interval)
        f_low = 1 / high_period  # Convert period to frequency
        f_high = 1 / low_period  # Convert period to frequency

        # Create a bandpass filter mask
        bandpass_mask = (np.abs(frequencies) >= f_low) & (np.abs(frequencies) <= f_high)

        # Apply the mask to the FFT data
        filtered_fft_data = fft_data * bandpass_mask

        # Apply inverse FFT to get the filtered time series back in the time domain
        filtered_time_series = np.fft.ifft(filtered_fft_data).real

        # Store the filtered time series back in the corresponding array slice
        filtered_data[:, i] = filtered_time_series

    return filtered_data


def bandpass_filter_via_fft_3d(data, low_period, high_period, sampling_interval=1):
    """
    Apply a bandpass filter to a 3D time series (time, lat, lon) using FFT and inverse FFT, optimized for vectorized operations.
    
    Parameters:
    - data: numpy array, the input time series data of shape (time, lat, lon).
    - low_period: float, the lower bound of the period (in days) to allow through the filter.
    - high_period: float, the upper bound of the period (in days) to allow through the filter.
    - sampling_interval: float, the sampling interval of the time series (default is 1 day).
    
    Returns:
    - filtered_data: numpy array, the bandpass-filtered time series.
    """
    # Perform FFT on the entire data array along the time axis
    fft_data = np.fft.fft(data, axis=0)
    
    # Generate frequency axis and define bandpass frequency limits
    frequencies = np.fft.fftfreq(data.shape[0], d=sampling_interval)
    f_low = 1 / high_period  # Convert period to frequency
    f_high = 1 / low_period  # Convert period to frequency

    # Create a bandpass filter mask
    bandpass_mask = (np.abs(frequencies) >= f_low) & (np.abs(frequencies) <= f_high)

    # Apply the mask to the FFT data
    filtered_fft_data = fft_data * bandpass_mask[:, np.newaxis, np.newaxis]

    # Apply inverse FFT to get the filtered time series back in the time domain
    filtered_data = np.fft.ifft(filtered_fft_data, axis=0).real

    return filtered_data



def detrend(data):
    """
    Detrend a numpy array by removing the best fit line.

    Args:
    data (np.array): The input data array (1D).

    Returns:
    np.array: The detrended data.
    """
    # Get the indices (x values)
    x = np.arange(len(data))
    
    # Fit a linear trend (y = mx + c)
    m, c = np.polyfit(x, data, 1)  # `1` here means linear (first order polynomial)
    
    # Calculate the trend as mx + c
    trend = m * x + c
    
    # Subtract the trend from the original data
    detrended_data = data - trend
    
    return detrended_data



def generate_time_series(cycles, years_do):
    """
    Generates a time series extended over multiple years with specified sinusoidal cycles.
    
    Parameters:
    - cycles: List of frequencies for the sinusoidal components.
    - years_do: Number of years to replicate the daily data.
    
    Returns:
    - extended_data: Numpy array of the extended time series data.
    """
    # Create daily data spanning one year
    xdaily = np.linspace(0, 2*np.pi*years_do, 365*years_do)
    # Initialize the array to hold the sum of sinusoidal signals
    signals = np.zeros_like(xdaily)
    
    # Add sinusoidal cycles specified by cycles list
    for cc in cycles:
        signals += np.sin(cc * (xdaily))
    
    # # Tile the combined signal to cover multiple years
    # extended_data = np.tile(signals, years_do)
    
    return signals

# Butterworth Bandpass Filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def lagged_correlation(x, y, maxlag):
    """Compute lagged correlations between two pandas.Series."""
    correlations = {}
    for lag in range(-maxlag, maxlag + 1):
        if lag >= 0:
            correlations[lag] = x.corr(y.shift(lag))
        else:
            correlations[lag] = x.shift(-lag).corr(y)
    return correlations


def regrid_(inBIG,method='bilinear',level=None):
    """
    Regrid the input variable from a list of files to a predefined grid.

    Parameters:
    - file_list (list): List of input files.
    - varin (str): Input variable name.
    - varout (str): Output variable name.
    - method (str): Resampling method (default is 'bilinear').
    - level (str, optional): Vertical level to select (default is None).

    Returns:
    - xr.Dataset: Regridded dataset.
    """
        
    ds_out = xr.open_dataset('/glade/work/wchapman/miles-rollout/notebooks/gather_global_data/ML_1deg_grid.nc')
    fn = xr.open_dataset('/glade/work/wchapman/miles-rollout/notebooks/gather_global_data/bilinear_640x1280_192x288.nc')
    regridder = xe.Regridder(inBIG, ds_out, "bilinear",weights=fn)
    outSMALL = regridder(inBIG)
    return outSMALL


def check_MJO_orientation(eof_list, pcs, lons, scaling_factors = None):
    """
    Check the orientation of MJO's first two Empirical Orthogonal Functions (EOFs).

    Parameters:
        eof_list (list): List of empirical orthogonal functions (EOFs).
        pcs (xarray.Dataset): Xarray dataset containing the principal components (PCs).
        lons (array_like): Longitudes.

    Returns:
        loc1 (int): Index of the first dominant EOF.
        loc2 (int): Index of the second dominant EOF.
        scale1 (int): Scaling factor for the first dominant EOF.
        scale2 (int): Scaling factor for the second dominant EOF.
    """
    bingo = 1
    if scaling_factors is not None:
        loc1 = scaling_factors[0]
        loc2 = scaling_factors[1]
        scale1 = scaling_factors[2]
        scale2 = scaling_factors[3]
        return loc1, loc2, scale1, scale2
    else:    
        # Get the first and second EOFs for OLR
        eof1_olr = eof_list[0][0, :]
        eof2_olr = eof_list[0][1, :]
        #print('len lons:',len(lons))
        #print(np.max(np.abs(eof1_olr)),'max abs1!')
        #print(np.max(np.abs(eof2_olr)),'max abs2!')
        # Find the longitude indices of maximum values for the first and second EOFs
        maxolr1_loc = int(np.where(np.abs(eof1_olr.squeeze()) == np.max(np.abs(eof1_olr)))[0][0])
        # print(maxolr1_loc, lons[maxolr1_loc])
        maxolr2_loc = int(np.where(np.abs(eof2_olr.squeeze()) == np.max(np.abs(eof2_olr)))[0][0])
        # print(maxolr2_loc, lons[maxolr2_loc])
        #print(maxolr1_loc,'loc max abs1!')
        #print(maxolr2_loc,'loc max abs2!')
        #print(np.abs(eof1_olr.squeeze()),'checkthis')

        # Check the orientation of MJO's first two EOFs
        if maxolr1_loc > maxolr2_loc:
            loc1 = 0
            loc2 = 1
        else:
            loc1 = 1
            loc2 = 0

        # Determine the scaling factors for the first two EOFs based on their signs

        if loc1 == 0:
            if eof1_olr[maxolr1_loc] > 0:
                scale1 = 1
            else:
                scale1 = -1

            if eof2_olr[maxolr2_loc] > 0:
                scale2 = -1
            else:
                scale2 = 1

        elif loc1 == 1:
            if eof1_olr[maxolr1_loc] > 0:
                scale1 = -1
            else:
                scale1 = 1

            if eof2_olr[maxolr2_loc] > 0:
                scale2 = 1
            else:
                scale2 = -1


        loc1 = loc1
        loc2 = loc2 
        scale1 = scale1
        scale2 = scale2 
        return loc1, loc2, scale1, scale2

def plot_obs_eof(eof_list, pcs, varfrac, lons, save_Fig_out, scaling_factors):
        """
        Plot the spatial structures of EOF1, EOF2, and EOF3.

        Parameters:
            eof_list (list): List of EOF arrays for OLR, U850, and U200.
            pcs (numpy.array): Array containing principal components.
            varfrac (numpy.array): Array containing the variance fraction values for each EOF.
            lons (numpy.array): Array of longitudes.

        Returns:
            None
        """

        loc1, loc2, scale1, scale2 =  scaling_factors[0],scaling_factors[1],scaling_factors[2],scaling_factors[3]
        print(f'Current Scaling Factors are:  {loc1}, {loc2}, {scale1}, {scale2}')

        # Calculate scaled EOF arrays and principal components
        eof1_olr = eof_list[0][loc1, :] * scale1
        eof2_olr = eof_list[0][loc2, :] * scale2
        eof3_olr = eof_list[0][2, :]

        eof1_u850 = eof_list[1][loc1, :] * scale1
        eof2_u850 = eof_list[1][loc2, :] * scale2
        eof3_u850 = eof_list[1][2, :]

        eof1_u200 = eof_list[2][loc1, :] * scale1
        eof2_u200 = eof_list[2][loc2, :] * scale2
        eof3_u200 = eof_list[2][2, :]

        pc1 = pcs[:, loc1] * scale1
        pc2 = pcs[:, loc2] * scale2
        pc3 = pcs[:, 2] * 1

        # Plot EOF1
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('Spatial structures of EOF1 and EOF2', fontsize=14)

        ax = fig.add_subplot(211)
        plt.title('EOF1 (' + str(int(varfrac[loc1] * 100)) + '%)', fontsize=10)
        plt.plot(lons, eof1_olr, color='k', linewidth=2, linestyle='solid', label='OLR')
        plt.plot(lons, eof1_u850, color='r', linewidth=2, linestyle='dashed', label='U850')
        plt.plot(lons, eof1_u200, color='b', linewidth=2, linestyle='dotted', label='U200')
        plt.axhline(0, color='k')
        plot_eof(ax, lons)

        # Plot EOF2
        ax = fig.add_subplot(212)
        plt.title('EOF2 (' + str(int(varfrac[loc2] * 100)) + '%)', fontsize=10)
        plt.plot(lons, eof2_olr, color='k', linewidth=2, linestyle='solid', label='OLR')
        plt.plot(lons, eof2_u850, color='r', linewidth=2, linestyle='dashed', label='U850')
        plt.plot(lons, eof2_u200, color='b', linewidth=2, linestyle='dotted', label='U200')
        plt.axhline(0, color='k')
        plot_eof(ax, lons)

        # # Plot EOF3
        # ax = fig.add_subplot(313)
        # plt.title('EOF3 (' + str(int(varfrac[2] * 100)) + '%)', fontsize=10)
        # plt.plot(lons, eof3_olr, color='k', linewidth=2, linestyle='solid', label='OLR')
        # plt.plot(lons, eof3_u850, color='r', linewidth=2, linestyle='dashed', label='U850')
        # plt.plot(lons, eof3_u200, color='b', linewidth=2, linestyle='dotted', label='U200')
        # plt.axhline(0, color='k')
        # plot_eof(ax, lons)

        # Save the plot as an image
        plt.savefig(save_Fig_out, bbox_inches='tight', dpi=400)
        plt.show()

def plot_eof(ax,lons):
    """
    Customize the EOF plot axis.

    Parameters:
        ax (matplotlib.axes.Axes): The axis of the plot to customize.

    Returns:
        ax (matplotlib.axes.Axes): The customized axis of the plot.
    """
    
    lon_formatter = LongitudeFormatter(number_format='.0f')

    plt.xlabel('Longitude')
    ax.tick_params(labelsize=10) 
    ax.xaxis.set_major_formatter(lon_formatter)
    plt.legend()

    return ax

def get_phase_and_eofs(eof_list, pcs, lons, scaling_factors):
    """
    Calculate MJO phase and related EOFs.

    Parameters:
        eof_list (list): List of empirical orthogonal functions (EOFs).
        pcs (xarray.Dataset): Xarray dataset containing the principal components (PCs).
        lons (array_like): Longitudes.

    Returns:
        tot_dict (dict): Dictionary containing MJO phase, RMM indices, and EOFs.
    """

    # Get the locations and scales for MJO EOF1 and EOF2
    loc1, loc2, scale1, scale2 = scaling_factors[0],scaling_factors[1],scaling_factors[2],scaling_factors[3]

    # Calculate scaled EOFs for OLR, U850, and U200 for MJO EOF1 and EOF2
    eof1_olr = eof_list[0][loc1, :] * scale1
    eof2_olr = eof_list[0][loc2, :] * scale2
    eof3_olr = eof_list[0][2, :]

    eof1_u850 = eof_list[1][loc1, :] * scale1
    eof2_u850 = eof_list[1][loc2, :] * scale2
    eof3_u850 = eof_list[1][2, :]

    eof1_u200 = eof_list[2][loc1, :] * scale1
    eof2_u200 = eof_list[2][loc2, :] * scale2
    eof3_u200 = eof_list[2][2, :]

    # Calculate scaled principal components (RMM indices) for MJO EOF1 and EOF2
    pc1 = pcs[:, loc1] * scale1
    pc2 = pcs[:, loc2] * scale2

    # Calculate RMM indices (RMM1_obs and RMM2_obs) based on scaled PCs
    RMM1_obs = pc1
    RMM2_obs = pc2

    # Calculate MJO phase based on RMM indices
    MJO_phase = []
    RMMind = np.sqrt(RMM1_obs**2 + RMM2_obs**2)  # Full index

    for ii in range(RMMind.shape[0]):    
        if np.isnan(RMMind[ii]):
            MJO_phase.append(np.nan)
        elif RMMind[ii] < 1:
            MJO_phase.append(0)
        else:
            ang = np.degrees(np.arctan2(RMM2_obs[ii], RMM1_obs[ii]))
            if ang < 0:
                ang = ang + 360
            ang = ang + 180

            if ang > 360:
                ang = ang - 360

            MJO_phase.append(np.floor((ang) / 45) + 1)

    MJO_phase = np.array(MJO_phase)

    # Create a dictionary to store calculated MJO phase, RMM indices, and EOFs
    tot_dict = {
        'MJO_phase': MJO_phase,
        'RMM1_obs': RMM1_obs,
        'RMM2_obs': RMM2_obs,
        'RMMind': RMMind,
        'eof1_olr': eof1_olr,
        'eof2_olr': eof2_olr,
        'eof1_u200': eof1_u200,
        'eof2_u200': eof2_u200,
        'eof1_u850': eof1_u850,
        'eof2_u850': eof2_u850
    }

    return tot_dict

def save_out_obs(tot_dict, u200, u850, olr, save_nc_out):
    """
    Save the observed MJO dataset to a NetCDF file.

    Parameters:
        tot_dict (dict): Dictionary containing MJO phase, RMM indices, and EOFs.
        u200 (xarray.DataArray): Xarray DataArray containing normalized u200 data.
        u850 (xarray.DataArray): Xarray DataArray containing normalized u850 data.
        olr (xarray.DataArray): Xarray DataArray containing normalized OLR data.

    Returns:
        MJO_fobs (xarray.Dataset): Xarray dataset containing the observed MJO data.
    """

    # Create a new xarray dataset to store the observed MJO data
    MJO_fobs = xr.Dataset(
        {
            "RMM1_obs": (["time"], tot_dict['RMM1_obs']),
            "RMM2_obs": (["time"], tot_dict['RMM2_obs']),
            "RMMind_obs": (["time"], tot_dict['RMMind']),
            "RMMphase_obs": (["time"], tot_dict['MJO_phase']),
            "olr_norm": (["time", "lon"], olr.data),
            "eof1_olr": (["lon"], tot_dict['eof1_olr']),
            "eof2_olr": (["lon"], tot_dict['eof2_olr']),
            "eof1_u850": (["lon"], tot_dict['eof1_u850']),
            "eof2_u850": (["lon"], tot_dict['eof2_u850']),
            "eof1_u200": (["lon"], tot_dict['eof1_u200']),
            "eof2_u200": (["lon"], tot_dict['eof2_u200']),
            "u200_norm": (["time", "lon"], u200.data),
            "u850_norm": (["time", "lon"], u850.data),
        },
        coords={
            "time": olr.time,
            "lon": olr.lon.values,
        },
    )

    # Add attributes to the dataset
    MJO_fobs.attrs["title"] = "MJO RMM Forecast eof(u850,u200,olr)"
    MJO_fobs.attrs["description"] = "MJO obs in the dataset calculated as in Wheeler and Hendon 2004, a 120-day filter, 15S-15N averaged variables "
    MJO_fobs.attrs["author"] = "S2S_WH_MJO_Forecast_Research_Toolbox"
    MJO_fobs.attrs["questions"] = "wchapman@ucar.edu"

    MJO_fobs.RMM1_obs.attrs['units'] = 'stddev'
    MJO_fobs.RMM1_obs.attrs['standard_name'] = 'RMM1'
    MJO_fobs.RMM1_obs.attrs['long_name'] = 'RMM1'

    MJO_fobs.RMM2_obs.attrs['units'] = 'stddev'
    MJO_fobs.RMM2_obs.attrs['standard_name'] = 'RMM2'
    MJO_fobs.RMM2_obs.attrs['long_name'] = 'RMM2'

    MJO_fobs.olr_norm.attrs['units'] = 'stddev'
    MJO_fobs.olr_norm.attrs['standard_name'] = 'outgoing longwave - normalized'
    MJO_fobs.olr_norm.attrs['long_name'] = 'outgoing longwave - normalized'

    MJO_fobs.u200_norm.attrs['units'] = 'stddev'
    MJO_fobs.u200_norm.attrs['standard_name'] = 'u200 normalized'
    MJO_fobs.u200_norm.attrs['long_name'] = 'u200 normalized'

    MJO_fobs.u850_norm.attrs['units'] = 'stddev'
    MJO_fobs.u850_norm.attrs['standard_name'] = 'u200 normalized'
    MJO_fobs.u850_norm.attrs['long_name'] = 'u200 normalized'

    # Save the MJO dataset to a NetCDF file
    MJO_fobs.to_netcdf(save_nc_out)
    print('...saved out ...')

    return MJO_fobs


# Example usage:
# Assuming 'data' is a numpy array of shape (time, lat, lon) representing your time series data
# low_period = 100 days, high_period = 20 days, sampling_interval = 1 day (default)