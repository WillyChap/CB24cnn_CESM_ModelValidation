#!/bin/bash

module load conda
conda activate npl-2023b

# Define the base directory for input files
DIR_IN="/glade/derecho/scratch/wchapman/ADF/f.e.FTORCHmjo_meanGPU_exp001/ts/"

# Array of filenames and corresponding variable names
declare -a FILES=(
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.V850.1979010100000-1993123100000.nc V850"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.V200.1979010100000-1993123100000.nc V200"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.U850.1979010100000-1993123100000.nc U850"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.U200.1979010100000-1993123100000.nc U200"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.FLUT.1979010100000-1993123100000.nc FLUT"
    "f.e.FTORCHmjo_meanGPU_exp001.cam.h1.PRECT.1979010100000-1993123100000.nc PRECT"
)

# Loop through the array and execute the Python script for each file and variable
for item in "${FILES[@]}"; do
    # Split item into filename and variable name
    read -r f_in var_name <<< "$item"
    # Execute the Python script
    python 01_Make_Climo.py --dir_in "${DIR_IN}" --f_in "${f_in}" --var_name "${var_name}"
done
