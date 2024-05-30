#!/bin/bash
#PBS -N MakeClimo
#PBS -A NAML0001
#PBS -l walltime=06:00:00
#PBS -o MakeClimo_f.e.FTORCHmjo_fullCNN_DT.out
#PBS -e MakeClimo_f.e.FTORCHmjo_fullCNN_DT.out
#PBS -q casper
#PBS -l select=1:ncpus=5:mem=210GB
#PBS -m a
#PBS -M wchapman@ucar.edu

module load conda
conda activate /glade/u/apps/opt/conda/envs/npl-2023b

# Define the base directory for input files

# DIR_IN="/glade/derecho/scratch/wchapman/ADF/ERA5_data/ts/"
# #Array of filenames and corresponding variable names
# declare -a FILES=(
#     "ERA5.h1.umag10.1979010100000-1993123100000.nc umag10"
# )

DIR_IN="/glade/derecho/scratch/wchapman/ADF/f.e.FTORCHmjo_fullCNN_DT//ts/"
#Array of filenames and corresponding variable names
declare -a FILES=(
    "f.e.FTORCHmjo_fullCNN_DT.cam.h1.Q.1979010100000-1990123100000.nc Q"
)

# Loop through the array and execute the Python script for each file and variable
for item in "${FILES[@]}"; do
    # Split item into filename and variable name
    read -r f_in var_name <<< "$item"
    # Execute the Python script
    python 01_Make_Climo.py --dir_in "${DIR_IN}" --f_in "${f_in}" --var_name "${var_name}"
done
