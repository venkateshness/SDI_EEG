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
    │   └── rest
    ├── Results/
    │   ├── Figure1/
    │   ├── ...
    │   └── Figure7/
    ├── src_scripts
    ├── .gitignore
    ├── readme.md
    └── requirements.txt
```

This repo contains `src_scripts`

## Dataset Downloader
We analyse subset of the data acquired by Healthy Brain Network (HBN). `_2_Downloading_from_AWS.sh` downloads the data necessary. Run them as `sh _2_Downloading_from_AWS.sh <sub_list>`. The `<sub_list>` can be obtained by running `_1_parsing_for_subjects.py`.

## Preprocessing and Source Localization
Once the dataset is downloaded, next step is loading the dataset, perform preprocessing and export them into MNE datastructure. To do so, `_3_loading_datasets.py` needs to be run. Follow the folder structure provided below for a painless usage. Sources of the scalp-level EEG is performed using eLORETA + BEM, implemented in MNE-Python. `_4_Source_Inversion_movie.py`, `_4_Source_Inversion_anvideo.py`, and `_4_Source_Inversion_rest.py` contain the full-pipeline for the source inversion. 

## Structural Connectome construction
Please refer to the guide at https://hackmd.io/@venki159/BJ1RGHbJp for a detailed procedure leveraging Qsiprep and Freesurfer to run in HPCs such as Compute Canada. 

## Structure-Function relationship
`_5_SDI_with_util_functions_individual_graphs.py` computes SDI for the individual graphs, and the `_5_SDI_with_util_functions_consensus_graphs.py` for the group-specific graphs. Both use the utility functions from `utility_functions.py`. Related stats procedure can be found in `_8_SDI_statistics.py`. 

