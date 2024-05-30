[![DOI](https://zenodo.org/badge/808131207.svg)](https://zenodo.org/doi/10.5281/zenodo.11395772)


# Repository to Create All Figures in:
## **_A State-Dependent Model-Error Representation for Online Climate Model Bias Correction_**

**Authors:**
- **Will Chapman** (NSF - NCAR) [wchapman@ucar.edu](mailto:wchapman@ucar.edu)
- **Judith Berner** (NSF - NCAR) [berner@ucar.edu](mailto:berner@ucar.edu)

### File Directory Path:

**Figure 01:**
- `.-|-> ./Skill_Network/Skill_CNN_Vertical_roll_anom.ipynb`

**Figure 02:**
- `.-|-> ./Vert_RMSE/Make_BS_Figure_Final.ipynb`

**Figure 03:**
- `.-|-> ./Modes_of_Variability/Panel_Plot.ipynb`

**Figure 04:**
- `.-|-> ./CMJO_Diagnostic_Tool/Panel_OLR_figure.ipynb`

### Supplemental Figures:

**Wheeler-Kiladis:**
- `.-|-> ./CMJO_Diagnostic_Tool/wk_spectra/*`

**Bias Panel Plots:**
- `.-|-> ./Climo_Bias_Tiles/CLIMO_Tiles.ipynb`

**OMEGA Bias:**
- `.-|-> ./Omega/OMEGA_WALKER.ipynb`

### Abstract

In this study, we develop a novel approach to correct biases in the atmospheric component of the Community Earth System Model (CESM) using convolutional neural networks (CNNs) to create a corrective model parameterization for online reduction. By learning to predict systematic nudging increments derived from a linear relaxation towards the ERA5 reanalysis, our method dynamically adjusts the model state, significantly outperforming traditional corrections based on climatological increments alone. Our results demonstrate substantial improvements in the root mean square error (RMSE) across all state variables, with precipitation biases over land reduced by 25-35%, depending on the season. Beyond reducing climate biases, our approach enhances the representation of major modes of variability, including the North Atlantic Oscillation (NAO) and other key aspects of boreal winter variability. A particularly notable improvement is observed in the Madden-Julian Oscillation (MJO), where the CNN-corrected model successfully propagates the MJO across the maritime continent, a challenge for many current climate models. This advancement underscores the potential of using CNNs for real-time model correction, providing a robust framework for improving climate simulations. Our findings highlight the efficacy of integrating machine learning techniques with traditional dynamical models to enhance climate prediction accuracy and reliability. This hybrid approach offers a promising direction for future research and operational climate forecasting, bridging the gap between observed and simulated climate dynamics.
