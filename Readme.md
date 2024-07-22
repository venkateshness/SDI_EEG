## Structure-function coupling and decoupling during movie-watching and resting-state: Novel insights bridging EEG and structural imaging

Briefly, this work contributes to understanding more about the relationship of the continuous EEG (Movie and Rest) on the underlying structure. 

Preprint: http://biorxiv.org/content/10.1101/2024.04.05.588337v1

In what follows, the overall structure of the code, and the data is described to be able to reproduce the results

Project is organized as follows:
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
    ├── src_data/
    │   ├── video1
    │   ├── video2
    │   ├── rest
    │   └── ....
    ├── src_scripts/
    │   ├── ....
    │   ├── ....
    │   ├── ....
    │   └── ....
    ├── .gitignore
    ├── readme.md
    └── requirements.txt
```

`Generated_data` and `src_data` can be found in https://osf.io/fme6x/. `Generated_data` are the intermediate data generated as part of the analysis. They are organized into first video, resting-state, and the second video. Each of them contains preprocessed scalp-level EEG (`preprocessed_dataset/`), source-localized EEG (`cortical_surface_related/`), and SDI-related (see the paper for more). `src_data` contains the data necessary for the analysis. More details about these folders are described in the OSF repo.

## Dataset Downloader
We analyse subset of the data acquired by Healthy Brain Network (HBN). `_2_Downloading_from_AWS.sh` downloads the data necessary. Run them as `sh _2_Downloading_from_AWS.sh <sub_list>`. The `<sub_list>` can be obtained by running `_1_parsing_for_subjects.py`.

## Preprocessing and Source Localization
Once the dataset is downloaded, next step is loading the dataset, perform preprocessing and export them into MNE datastructure. To do so, `_3_loading_datasets.py` needs to be run. Follow the folder structure provided above for a hassle-free usage. Sources of the scalp-level EEG are estimated using eLORETA + BEM, implemented in MNE-Python. `_4_Source_Inversion_video1.py`, `_4_Source_Inversion_video2.py`, and `_4_Source_Inversion_rest.py` contain the full-pipeline for the source inversion. 

## Structural Connectome construction
Please refer to the guide at https://hackmd.io/@venki159/BJ1RGHbJp (to be updated) for a detailed procedure leveraging Qsiprep and Freesurfer to run in HPCs such as Compute Canada. 

## Structure-Function relationship
`_5_SDI_with_util_functions_individual_graphs.py` computes SDI with the individual graphs. It uses the utility functions from `utility_functions.py`. Related stats procedure can be found in `_6_SDI_statistics.py`. `_9_ICC_reliability_check.py` tests for the reliability by computing the Intra-class coefficient (ICC) between Video 1 and Video 2.

## Figures Generation
Figures are generated through two scripts. `8_figures_generation.py` is the OG file that does two jobs: a:generates figures that are not spatial maps-generated; b: exports data for generating the spatial maps for `8_figures_generation_spatial_maps.py`

## Miscellaneous 
Customised `plotting_img_on_surf` function of Nilearn is in `_7_SDI_spatial_maps.py`, which comes in handy to visualize the spatial maps on the go.

The above scripts are for the main analysis. What follows are for the analysis described in the supplementary material

## Inter-subject correlation
`_10_ISC.py` computes the correlation of cortical activity among subjects using Correlated Component Analysis (CorrCA). This script makes use of the utility script `util_5_CorrCA.py`. Once estimated, SDI is computed specifically for certain segments in the video and the related script is in `_10_SDI_Strongest_ISC.py`.

Additionally, SDI compared across frequency bands at the macro-level (Yeo-Krienen networks), script is in `yeo_krienen_network.py`


INFO 1: Decoding script runs on older version of NiMare (0.0.14). Latest version requires Pandas 2.2.0, for which some part of the code is deprecated. I recommend to create a new environment for NiMare 0.0.14. Analysis pipeline + Decoding are guaranteed to run without any trouble.

INFO 2: `src_data/` in the OSF repo does not contain the raw Healthy Brain Network (HBN) data, can be downloaded using `_2_Downloading_from_AWS.sh`

HBN imaging and EEG data are shared under the Creative Commons Attribution 4.0 International License (CC-BY-4.0). The shared data also includes preprocessed and generated data applicable to this study.