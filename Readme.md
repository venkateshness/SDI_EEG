## Title to come
Briefly, this work contributes to understanding more about the relationship of the continuous EEG (Movie and Rest) on the underlying structure. 

```
└── structure-function-eeg/ 
    ├── Generated_data  / 
    │   ├── video1/
    │   │   ├── cortical_surface_related
    │   │   ├── SDI_related
    │   │   └── preprocessed_dataset
    │   ├── video2/
    │   │   ├── cortical_surface_related
    │   │   ├── SDI_related
    │   │   └── preprocessed_dataset
    │   ├── rest/
    │   │   ├── cortical_surface_related
    │   │   ├── SDI_related
    │   │   └── preprocessed_dataset
    │   ├── video1_consensus/
    │   │   └── SDI_related
    │   └── video2_consensus/
    │       └── SDI_related
    ├── src_data/
    │   ├── video1
    │   ├── video2
    │   ├── rest
    │   └── ....
    ├── Results/
    │   ├── Figure1/
    │   ├── ...
    │   └── Figure7/
    ├── src_scripts
    ├── .gitignore
    ├── readme.md
    └── requirements.txt
```

This repo contains all the code-related files from `src_scripts`. `src_data` and `Generated_data` can be found in https://osf.io/rjuap/. `src_data` contains the source/raw data necessary for the analysis. `Generated_data` are the intermediate data generated during analysis, and are used for subsequent steps. Code for selecting the subjects to downloading the data to generating final figures are provided below.

## Dataset Downloader
We analyse subset of the data acquired by Healthy Brain Network (HBN). `_2_Downloading_from_AWS.sh` downloads the data necessary. Run them as `sh _2_Downloading_from_AWS.sh <sub_list>`. The `<sub_list>` can be obtained by running `_1_parsing_for_subjects.py`.

## Preprocessing and Source Localization
Once the dataset is downloaded, next step is loading the dataset, perform preprocessing and export them into MNE datastructure. To do so, `_3_loading_datasets.py` needs to be run. Follow the folder structure provided above for a hassle-free usage. Sources of the scalp-level EEG are estimated using eLORETA + BEM, implemented in MNE-Python. `_4_Source_Inversion_video1.py`, `_4_Source_Inversion_video2.py`, and `_4_Source_Inversion_rest.py` contain the full-pipeline for the source inversion. 

## Structural Connectome construction
Please refer to the guide at https://hackmd.io/@venki159/BJ1RGHbJp for a detailed procedure leveraging Qsiprep and Freesurfer to run in HPCs such as Compute Canada. 

## Structure-Function relationship
`_5_SDI_with_util_functions_individual_graphs.py` computes SDI with the individual graphs, and the `_5_SDI_with_util_functions_consensus_graphs.py` for the group-specific consensus graphs. Both use the utility functions from `utility_functions.py`. Related stats procedure can be found in `_8_SDI_statistics.py`. 

The above scripts are for the main analysis. What follows are for the analysis described in the supplementary material
## Inter-subject correlation
`_10_ISC.py` computes the correlation of cortical activity among subjects using Correlated Component Analysis (CorrCA). This script makes use of the utility script `util_5_CorrCA.py`. Once estimated, SDI is computed specifically for certain segments in the video and the related script is in `_10_SDI_Strongest_ISC.py`.

Additionally, SDI compared across frequency bands at the macro-level (Yeo-Krienen networks), script is in `yeo_krienen_network.py`