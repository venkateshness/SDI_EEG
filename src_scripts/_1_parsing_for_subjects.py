#%%
import scipy
from scipy import io
import numpy as np
import pandas as pd
import ezodf
import os

HOMEDIR = "/users/local/Venkatesh/structure-function-eeg/"  # os.path.abspath(os.getcwd())

mat = scipy.io.loadmat(
    f"{HOMEDIR}/src_data/subject_inclusion/subject_list.mat")
df = pd.DataFrame(mat["good_EEG"])

df_sliced = [df.values[i][0][0] for i in range(len(df))]

df_SI = pd.read_csv(
    f"{HOMEDIR}/src_data/subject_inclusion/participants_SI.tsv")
df_RU = pd.read_csv(
    f"{HOMEDIR}/src_data/subject_inclusion/participants_RU.tsv", sep="\t")
df_CUNY = pd.read_csv(
    f"{HOMEDIR}/src_data/subject_inclusion/participants_CUNY.tsv", sep="\t")
df_CBIC = pd.read_csv(
    f"{HOMEDIR}/src_data/subject_inclusion/participants_CBIC.tsv", sep="\t")

subjects_aged = list()


def find_subject_age(which_df, age):
    """Filtering subject list based on their age

    Args:
        which_df (dataframe): the df from different HBN sites containing the totality of subj info
        age (_type_): threshold to apply filter for
    """
    
    subjects_aged.append(which_df[which_df["Age"] >= age]["participant_id"].values)


age = 14  # criteria 1; age
find_subject_age(df_CBIC, age)
find_subject_age(df_RU, age)
find_subject_age(df_CUNY, age)
find_subject_age(df_SI, age)
subjects_aged = np.hstack(subjects_aged)


def read_ods(filename, sheet_no=0, header=0):
    tab = ezodf.opendoc(filename=filename).sheets[sheet_no]
    return pd.DataFrame({
        col[header + 1].value: [x.value for x in col[header + 1:]]
        for col in tab.columns()
    })

# criteria 2; dwi
df_dwi = read_ods(f"{HOMEDIR}/src_data/subject_inclusion/dwi-subject_list.ods",
                  0, 0)
# criteria 3; freesurfer
df_freesurfer = read_ods(
    f"{HOMEDIR}/src_data/subject_inclusion/subjects_freesurfer.ods")



df_dwi_sliced = [
    df_dwi[df_dwi.columns[0]].values[i][31:] for i in range(0, len(df_dwi))
]

subjects_aged_sliced = [
    subjects_aged[i][4:] for i in range(0, len(subjects_aged))
]
total_subjects = list(
    set(df_dwi_sliced).intersection(
        set(df_sliced).intersection(set(subjects_aged_sliced))))

# criteria 4
# final_subject_list = list()
# for i in range(len(total_subjects)):
#     path_to_file = (
#         f"{HOMEDIR}/src_data/HBN_dataset/%s/RestingState_data.csv" %
#         total_subjects[i])
#     path_to_file_video = (
#         f"{HOMEDIR}/src_data/HBN_dataset/%s/Video3_event.csv" %
#         total_subjects[i])

#     if os.path.isfile(path_to_file) and os.path.isfile(path_to_file_video):
#         final_subject_list.append(total_subjects[i])

print("N = ", len(total_subjects))

with open(f"{HOMEDIR}/src_data/subjects_to_download_for.txt", "w") as output:
        out=total_subjects
        for row in range(len(out)):
            output.write(out[row]+'\n')

# %%
