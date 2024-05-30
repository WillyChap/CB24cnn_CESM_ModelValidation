#!/bin/bash
#PBS -N MakeLAG
#PBS -A NAML0001
#PBS -l walltime=03:00:00
#PBS -o XCorr_f.e.FTORCHmjo_CNNmjo_tot.out
#PBS -e XCorr_f.e.FTORCHmjo_CNNmjo_tot_data.out
#PBS -q casper
#PBS -l select=1:ncpus=10:mem=110GB
#PBS -m a
#PBS -M wchapman@ucar.edu

module load conda
conda activate /glade/u/apps/opt/conda/envs/npl-2023b

# Define the base directory for input files
DIR_IN="/glade/derecho/scratch/wchapman/ADF/f.e.FTORCHmjo_CNNmjo_tot/ts/climo"

# Array of filenames and corresponding variable names
declare -a FILES=(
    "f.e.FTORCHmjo_CNNmjo_tot.cam.h1.PRECT.anomalies.1979010100000-1991122700000.nc PRECT f.e.FTORCHmjo_CNNmjo_tot.cam.h1.TREFHT.anomalies.1979010100000-1991122700000.nc TREFHT"
    "f.e.FTORCHmjo_CNNmjo_tot.cam.h1.LHFLX.anomalies.1979010100000-1991122700000.nc LHFLX f.e.FTORCHmjo_CNNmjo_tot.cam.h1.U850.anomalies.1979010100000-1991122700000.nc U850"
    "f.e.FTORCHmjo_CNNmjo_tot.cam.h1.PRECT.anomalies.1979010100000-1991122700000.nc PRECT f.e.FTORCHmjo_CNNmjo_tot.cam.h1.U850.anomalies.1979010100000-1991122700000.nc U850"
    "f.e.FTORCHmjo_CNNmjo_tot.cam.h1.FLUT.anomalies.1979010100000-1991122700000.nc FLUT f.e.FTORCHmjo_CNNmjo_tot.cam.h1.U850.anomalies.1979010100000-1991122700000.nc U850"
)

# Loop through the array and execute the Python script for each file and variable
for item in "${FILES[@]}"; do
    # Split item into filename and variable name
    read -r f_precip_in var_name_precip f_u_in var_name_u <<< "$item"
    # Execute the Python script
    python 09_mjoxcor_lag_season.py --dir_in "${DIR_IN}" --f_precip_in "${f_precip_in}" --var_name_precip "${var_name_precip}" --f_u_in "${f_u_in}" --var_name_u "${var_name_u}"
done
