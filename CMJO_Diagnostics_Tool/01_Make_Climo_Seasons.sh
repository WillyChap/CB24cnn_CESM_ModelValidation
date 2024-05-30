#!/bin/bash

module load conda
conda activate /glade/u/apps/opt/conda/envs/npl-2023b

# Define the base directory for input files
DIR_IN="/glade/derecho/scratch/wchapman/ADF/ERA5_data/ts/"

# Array of filenames and corresponding variable names
declare -a FILES=(
    "GPCP_180_360_1997-2023.camgrid.nc precip 1979-01-15"
    "GPCP_180_360_1997-2023.camgrid.nc precip 1979-04-15"
    "GPCP_180_360_1997-2023.camgrid.nc precip 1979-07-15"
    "GPCP_180_360_1997-2023.camgrid.nc precip 1979-10-15"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.Z500.1979010100000-2010123100000.nc Z500 1979-01-15"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.Z500.1979010100000-2010123100000.nc Z500 1979-04-15"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.Z500.1979010100000-2010123100000.nc Z500 1979-07-15"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.Z500.1979010100000-2010123100000.nc Z500 1979-10-15"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.PRECT.1979010100000-2010123100000.nc PRECT 1979-01-15"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.PRECT.1979010100000-2010123100000.nc PRECT 1979-04-15"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.PRECT.1979010100000-2010123100000.nc PRECT 1979-07-15"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.PRECT.1979010100000-2010123100000.nc PRECT 1979-10-15"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.U200.1979010100000-2010123100000.nc U200 1979-01-15"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.U200.1979010100000-2010123100000.nc U200 1979-04-15"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.U200.1979010100000-2010123100000.nc U200 1979-07-15"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.U200.1979010100000-2010123100000.nc U200 1979-10-15"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.U850.1979010100000-2010123100000.nc U850 1979-01-15"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.U850.1979010100000-2010123100000.nc U850 1979-04-15"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.U850.1979010100000-2010123100000.nc U850 1979-07-15"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.U850.1979010100000-2010123100000.nc U850 1979-10-15"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.V200.1979010100000-2010123100000.nc V200 1979-01-15"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.V200.1979010100000-2010123100000.nc V200 1979-04-15"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.V200.1979010100000-2010123100000.nc V200 1979-07-15"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.V200.1979010100000-2010123100000.nc V200 1979-10-15"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.V850.1979010100000-2010123100000.nc V850 1979-01-15"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.V850.1979010100000-2010123100000.nc V850 1979-04-15"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.V850.1979010100000-2010123100000.nc V850 1979-07-15"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.V850.1979010100000-2010123100000.nc V850 1979-10-15"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.FLUT.1979010100000-2010123100000.nc FLUT 1979-01-15"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.FLUT.1979010100000-2010123100000.nc FLUT 1979-04-15"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.FLUT.1979010100000-2010123100000.nc FLUT 1979-07-15"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.FLUT.1979010100000-2010123100000.nc FLUT 1979-10-15"
)

# declare -a FILES=(
#     "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.PRECT.1979010100000-2010123100000.nc PRECT"
#     "f.e.FTORCHmjo_MEANmjo.cam.h1.FLUT.1979010100000-2010123100000.nc FLUT"
#     "f.e.FTORCHmjo_MEANmjo.cam.h1.U200.1979010100000-2010123100000.nc U200"
#     "f.e.FTORCHmjo_MEANmjo.cam.h1.V200.1979010100000-2010123100000.nc V200"
#     "f.e.FTORCHmjo_MEANmjo.cam.h1.U850.1979010100000-2010123100000.nc U850"
#     "f.e.FTORCHmjo_MEANmjo.cam.h1.V850.1979010100000-2010123100000.nc V850"
# )

# Loop through the array and execute the Python script for each file and variable
for item in "${FILES[@]}"; do
    # Split item into filename and variable name
    read -r f_in var_name date_in <<< "$item"
    # Execute the Python script
    python 01_Make_Climo_date.py --dir_in "${DIR_IN}" --f_in "${f_in}" --var_name "${var_name}" --date_in "${date_in}"
done
