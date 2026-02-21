#!/usr/bin/env python
# coding: utf-8
import os

# In[8]:


import numpy as np
#from sklearn.model_selection import train_test_split
from sktime.datasets import load_UCR_UEA_dataset
import pandas as pd
import sys


sys.path.append("../code")

from utils import medoid_per_class, export, medoid, export_univ_tmc
OUTPUT_FOLDER = 'forda'


# In[9]:


# Paso 1: Cargar FordA
X, y = load_UCR_UEA_dataset(name="FordA", return_X_y=True)
X = X.iloc[:, 0].apply(pd.Series)
y = np.where(y == '-1', int(0), int(1))  # Asegura etiquetas 0/1
y = pd.Series(y, index=X.index)
X.rename(lambda x: 'C'+str(x), axis=1, inplace=True)

# In[20]:


# Paso 2: Separar datos
medoid_ids_per_class = medoid_per_class(X, y)
df_only_num = X
df_num = df_only_num.copy()
df_num['Class'] = y

cols_items = [('C', 'C')]
cols_dict = dict(cols_items)
# In[ ]:
os.makedirs(f'{OUTPUT_FOLDER}/C', exist_ok=True)

for idx, row in df_only_num.iterrows():
    export_univ_tmc(row, cols_items, f'{idx:03d}_{int(df_num.iloc[idx]["Class"])}', OUTPUT_FOLDER)
# Suppose df1 has features + "Class"
for cls, group in df_num.groupby("Class"):
    export_univ_tmc(df_only_num.loc[medoid_ids_per_class[cls]], cols_items, f'{int(cls)}_medoid', OUTPUT_FOLDER)
    export_univ_tmc(df_only_num.loc[group.index].mean(axis=0), cols_items, f'{int(cls)}_centroid', OUTPUT_FOLDER)

global_centroid_VX = df_only_num.mean(axis=0)
export_univ_tmc(global_centroid_VX, cols_items, 'global_centroid', OUTPUT_FOLDER)

global_medoid_id = medoid(df_only_num)
export_univ_tmc(df_only_num.loc[global_medoid_id], cols_items, 'global_medoid', OUTPUT_FOLDER)



# In[10]:


metadata_dict = {}
global_medoid_id = medoid(df_only_num)

for (VAR, VARNAME) in cols_items:
    for idx, row in df_num.iterrows():
        nrow = row.copy()
        cls = int(row["Class"])
        filename = f'{OUTPUT_FOLDER}/{VARNAME}/{VARNAME}_{idx:03d}_{cls}.csv'
        metadata = {}
        metadata['series'] = filename
        metadata['class'] = cls
        metadata['predicted_class'] = cls
        metadata['predicted_class_probability'] = 0.6 if cls == 1 else 0.4
        metadata['channel'] = VARNAME.replace('_', ' ').capitalize()
        metadata['group'] = 0
        metadata['annotation'] = idx
        opposite_class = int(next(iter(set(df_num['Class'].unique()) - {row['Class']})))

        metadata['global_medoid_id'] = global_medoid_id
        cls_global_medoid = int(df_num.loc[global_medoid_id]['Class'])
        metadata['reference_1'] = f'{OUTPUT_FOLDER}/{VARNAME}/{VARNAME}_{global_medoid_id:03d}_{cls_global_medoid}.csv'
        metadata['reference_1_predicted_class'] = 1
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

        metadata['beta_attributions'] = f'{OUTPUT_FOLDER}/{VARNAME}/{VARNAME}_beta_inst_{idx:03d}.csv'
        metadata_dict[f'{VARNAME}' + str(idx)] = metadata


# In[11]:


meta_df = pd.DataFrame(metadata_dict)
meta_df.T.to_csv(f"{OUTPUT_FOLDER}/metadata.csv", index=False)


# In[12]:


metametadata_dict = {'units': {'C': 'dB'},
                     'channel-descriptions': {'C': 'Noise intensity'},
                     'class-descriptions': {'1': 'No problem', '0': 'Problem'},
                     'references': {'reference_1' : "Global Medoid", 'reference_2' : "Medoid of Opposite Class",
                                     'reference_3' : "Medoid of Opposite Class (Predicted)", 'reference_4' : "Global Centroid",
                                     'reference_5' : "Centroid of Opposite Class", 'reference_6' : "Centroid of Opposite Class (Predicted)"}}
import json
with open(f"{OUTPUT_FOLDER}/metametadata.json", mode="w", encoding="utf-8") as write_file:
    json.dump(metametadata_dict, write_file)


# In[ ]:







# In[ ]:




