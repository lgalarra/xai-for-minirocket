#!/usr/bin/env python
# coding: utf-8
import os

# In[13]:


import pandas as pd
import sys
sys.path.append("../code")

from utils import medoid_per_class, medoid

# Read the two parquet files
df1 = pd.read_hdf("cognitive-circles/df40participants.h5")
OUTPUT_FOLDER = 'cognitive-circles'


# In[14]:


import numpy as np
y = df1['RealDifficulty']
task = df1['Task']

df_only_num = df1.drop(df1.select_dtypes(exclude=[np.float64]), axis=1)
df_only_num = df_only_num.drop(columns=["ParticipantWantedToRerateTaskPerceivedDifficultyLevel"])

df_num = df_only_num.copy()
df_num['RealDifficulty'] = y
df_num['Task'] = task


# In[15]:

from utils import export
import numpy as np

cols_items = [('X', 'X'), ('V', 'velocity'), ('VA', 'angular_velocity'),
                           ('DR', 'radial_velocity'), ('Y', 'Y'), ('D', 'radius'),  ('A', 'acceleration')]
cols_dict = dict(cols_items)
medoid_ids_per_class = medoid_per_class(df_only_num, y)


# In[ ]:
for (VAR, VARNAME) in cols_items:
    os.makedirs(f'{OUTPUT_FOLDER}/{VARNAME}', exist_ok=True)

for (VAR, VARNAME) in cols_items:
    for idx, row in df_only_num.iterrows():
        export(row, cols_items, f'{idx:03d}_{df_num.iloc[idx]["RealDifficulty"]}', OUTPUT_FOLDER)
    # Suppose df1 has features + "Class"
    for cls, group in df_num.groupby("RealDifficulty"):
        export(df_only_num.loc[medoid_ids_per_class[cls]], cols_items, f'{cls}_medoid', OUTPUT_FOLDER)
        X = group.drop(columns=["RealDifficulty", "Task"])
        export(np.mean(X, axis=0), cols_items, f'{cls}_centroid', OUTPUT_FOLDER)
        for task, subgroup in group.groupby('Task'):
            X_subgroup = group.drop(columns=["RealDifficulty", "Task"])
            inst_id = medoid(X_subgroup)
            export(X_subgroup.loc[inst_id], cols_items, f'{cls}_{task}_{inst_id}_medoid', OUTPUT_FOLDER)
            centroid_task = np.mean(X_subgroup, axis=0)
            export(centroid_task, cols_items, f'{cls}_{task}_{inst_id}_centroid', OUTPUT_FOLDER)

    global_centroid_VX = np.mean(df_only_num, axis=0)
    export(global_centroid_VX, cols_items, 'global_centroid', OUTPUT_FOLDER)

global_medoid_id = medoid(df_only_num)
export(df_only_num.loc[global_medoid_id], cols_items, 'global_medoid', OUTPUT_FOLDER)



# In[10]:


metadata_dict = {}
global_medoid_id = medoid(df_only_num)

for (VAR, VARNAME) in cols_items:
    for idx, row in df1.iterrows():
        nrow = row.copy()
        cls = row["RealDifficulty"]
        filename = f'{OUTPUT_FOLDER}/{VARNAME}/{VARNAME}_{idx:03d}_{cls}.csv'
        metadata = {}
        metadata['series'] = filename
        metadata['class'] = cls
        metadata['predicted_class'] = cls
        metadata['channel'] = VARNAME.replace('_', ' ').capitalize()
        metadata['group'] = row['ParticipantID']
        metadata['annotation'] = row['Task'] + '-' + row['RealDifficulty']
        opposite_class = next(iter(set(df1['RealDifficulty'].unique()) - {row['RealDifficulty']}))

        metadata['global_medoid_id'] = global_medoid_id
        cls_global_medoid = df1.loc[global_medoid_id]['RealDifficulty']
        metadata['reference_1'] = f'{OUTPUT_FOLDER}/{VARNAME}/{VARNAME}_{global_medoid_id:03d}_{cls_global_medoid}.csv'
        metadata['reference_1_predicted_class'] = 'facil'
        metadata['reference_1_predicted_class_probability'] = 0.8

        metadata['medoid_id_opposite_class'] = medoid_ids_per_class[opposite_class]
        metadata['reference_2'] = f'{OUTPUT_FOLDER}/{VARNAME}/{VARNAME}_{metadata["medoid_id_opposite_class"]:03d}_{opposite_class}.csv'
        metadata['reference_2_predicted_class'] = opposite_class
        metadata['reference_2_predicted_class_probability'] = 0.5

        metadata['medoid_id_opposite_predicted_class'] = metadata['medoid_id_opposite_class']
        metadata['reference_3'] = metadata['reference_2']
        metadata['reference_3_predicted_class'] = opposite_class
        metadata['reference_3_predicted_class_probability'] = 0.5

        metadata['reference_4'] = f'{OUTPUT_FOLDER}/{VARNAME}/global_centroid.csv'
        metadata['reference_4_predicted_class'] = 0
        metadata['reference_4_predicted_class_probability'] = 0.1

        metadata['reference_5'] = f'{OUTPUT_FOLDER}/{VARNAME}/{VARNAME}_{opposite_class}_centroid.csv'
        metadata['reference_5_predicted_class'] = 0
        metadata['reference_5_predicted_class_probability'] = 0.8

        metadata['reference_6'] = metadata['reference_5']
        metadata['reference_6_predicted_class'] = 0
        metadata['reference_6_predicted_class_probability'] = 0.8

        metadata['beta_attributions'] = f'{OUTPUT_FOLDER}/{VARNAME}/beta_inst_{idx:03d}.csv'
        metadata_dict[f'{VARNAME}' + str(idx)] = metadata


# In[11]:


meta_df = pd.DataFrame(metadata_dict)
meta_df.T.to_csv(f"{OUTPUT_FOLDER}/metadata.csv", index=False)


# In[12]:


metametadata_dict = {'units' : {'X': 'pixel', 'Y': 'pixel', 'V': 'pixel/s', 'VA': 'radians/s^2', 'DR': 'radians/s',
                                'D': 'pixel', 'A': 'pixel/s^2'},
                     'class-descriptions' : {'facil': 'Easy', 'dificil': 'Difficult'},
                     'channel-descriptions' : {'X': 'X-coordinate', 'Y': 'Y-coordinate', 'V': 'Linear Velocity',
                                       'VA': 'Angular Velocity', 'DR': 'Radial Velocity', 'D': 'Radius', 'A': 'Acceleration'},
                     'references' : {'reference_1' : "Global Medoid", 'reference_2' : "Medoid of Opposite Class",
                                     'reference_3' : "Medoid of Opposite Class (Predicted)", 'reference_4' : "Global Centroid",
                                     'reference_5' : "Centroid of Opposite Class", 'reference_6' : "Centroid of Opposite Class (Predicted)"}}
import json
with open(f"{OUTPUT_FOLDER}/metametadata.json", mode="w", encoding="utf-8") as write_file:
    json.dump(metametadata_dict, write_file)


# In[ ]:




