#!/bin/bash -l
#PBS -N VertRMSE
#PBS -A NAML0001
#PBS -l walltime=05:00:00
#PBS -o VertRMSE.out
#PBS -e VertRMSE.out
#PBS -q main
#PBS -l select=1:ncpus=5:mem=110GB
#PBS -m a
#PBS -M wchapman@ucar.edu

module load conda
conda activate npl-2023b

# Define directory paths
dir_exp="/glade/derecho/scratch/wchapman/ADF/f.e.FTORCHmjo_MEANmjo/ts/"
dir_cntrl="/glade/derecho/scratch/wchapman/ADF/f.e.FTORCHmjo_CNTRLmjo/ts/"
dir_obs="/glade/campaign/cgd/amp/wchapman/Reanalysis/ERA5_obs/"

# Define common filename parts
prefix_exp="f.e.FTORCHmjo_MEANmjo"
prefix_cntrl="f.e.FTORCHmjo_CNTRLmjo"
suffix_exp=".cam.h0."
suffix_cntrl=".cam.h0."
suffix_obs="ERA5.an.sfc.pl.camgrid.1979-2022.nc"

# Define variables and their corresponding file part
declare -a variables=("T")
declare -a file_parts=("T.plev.197901-201012.nc")

# Loop through the variables and execute the Python script
for i in "${!variables[@]}"; do
    var="${variables[$i]}"
    file_part="${file_parts[$i]}"

    # Execute the Python script with the correct parameters
    python Vertical_RMSE_bs_tropics.py \
        --dir_exp "$dir_exp" \
        --dir_cntrl "$dir_cntrl" \
        --dir_obs "$dir_obs" \
        --fin_exp "${prefix_exp}${suffix_exp}${file_part}" \
        --fin_cntrl "${prefix_cntrl}${suffix_cntrl}${file_part}" \
        --fin_obs "$suffix_obs" \
        --var_cntrl "$var" \
        --var_exp "$var" \
        --var_obs "${var,,}"  # Assumes lowercase for observational data variable names
done
