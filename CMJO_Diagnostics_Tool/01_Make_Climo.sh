#!/bin/bash
#PBS -N MakeClimo
#PBS -A NAML0001
#PBS -l walltime=03:00:00
#PBS -o 01_MakeClimo_f.e.FTORCHmjo_CNTRLmjo_DT.out
#PBS -e 01_MakeClimo_f.e.FTORCHmjo_CNTRLmjo_DT.out
#PBS -q casper
#PBS -l select=1:ncpus=10:mem=110GB
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

DIR_IN="/glade/derecho/scratch/wchapman/ADF/f.e.FTORCHmjo_CNTRLmjo_DT/ts/"
#Array of filenames and corresponding variable names
declare -a FILES=(
    "f.e.FTORCHmjo_CNTRLmjo_DT.cam.h1.V850.1979010100000-1990122200000.nc V850"
    "f.e.FTORCHmjo_CNTRLmjo_DT.cam.h1.V200.1979010100000-1990122200000.nc V200"
    "f.e.FTORCHmjo_CNTRLmjo_DT.cam.h1.U850.1979010100000-1990122200000.nc U850"
    "f.e.FTORCHmjo_CNTRLmjo_DT.cam.h1.U200.1979010100000-1990122200000.nc U200"
    "f.e.FTORCHmjo_CNTRLmjo_DT.cam.h1.FLUT.1979010100000-1990122200000.nc FLUT"
    "f.e.FTORCHmjo_CNTRLmjo_DT.cam.h1.Z500.1979010100000-1990122200000.nc Z500"
    "f.e.FTORCHmjo_CNTRLmjo_DT.cam.h1.PRECT.1979010100000-1990122200000.nc PRECT"
    "f.e.FTORCHmjo_CNTRLmjo_DT.cam.h1.LHFLX.1979010100000-1990122200000.nc LHFLX"
    "f.e.FTORCHmjo_CNTRLmjo_DT.cam.h1.U10.1979010100000-1990122200000.nc U10"
    "f.e.FTORCHmjo_CNTRLmjo_DT.cam.h1.UBOT.1979010100000-1990122200000.nc UBOT"
    "f.e.FTORCHmjo_CNTRLmjo_DT.cam.h1.VBOT.1979010100000-1990122200000.nc VBOT"
    "f.e.FTORCHmjo_CNTRLmjo_DT.cam.h1.TBOT.1979010100000-1990122200000.nc TBOT"
    "f.e.FTORCHmjo_CNTRLmjo_DT.cam.h1.QREFHT.1979010100000-1990122200000.nc QREFHT"
    "f.e.FTORCHmjo_CNTRLmjo_DT.cam.h1.SHFLX.1979010100000-1990122200000.nc SHFLX"
    "f.e.FTORCHmjo_CNTRLmjo_DT.cam.h1.OMEGA500.1979010100000-1990122200000.nc OMEGA500"
)

# Loop through the array and execute the Python script for each file and variable
for item in "${FILES[@]}"; do
    # Split item into filename and variable name
    read -r f_in var_name <<< "$item"
    # Execute the Python script
    python 01_Make_Climo.py --dir_in "${DIR_IN}" --f_in "${f_in}" --var_name "${var_name}"
done
