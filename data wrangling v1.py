#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import numpy as np
import pandas as pd
import pyteomics
from pyteomics import tandem
import pymzml
import argparse
from pyteomics import mzml, auxiliary
from matplotlib import pyplot as plt


# In[3]:


try: 
    os.chdir('F://DS5500_project_1')
    print("Current directory is {}".format(os.getcwd()))
except: 
    print("Something wrong with specified directory. Exception- ", sys.exc_info())


# In[5]:


j = 0
total_spectrum_dict = {}
for filename in os.listdir(os.getcwd()): #iterate through all mzML file
    with mzml.read(os.path.join(os.getcwd(), filename), 'r') as reader:
        mzdict = {}
        intensitydict = {}
        i = 0
        for spec in reader:
            if (spec['ms level'] == 2):
                mzdict[i] = spec['m/z array'].tolist()
                intensitydict[i] = spec['intensity array'].tolist()
            i += 1
        spectrum_dict = {}
        for key in intensitydict.keys(): #create the raw data frame each of which represents a spectrum reading
            spectrum_dict[key] = pd.DataFrame({'location':mzdict[key],'intensity':intensitydict[key]})
        for key in spectrum_dict.keys(): #segment spectrums by intervals with length of 100
            spectrum_df = spectrum_dict[key]
            spectrum_df = spectrum_df.assign(interval=spectrum_df['location']//100+1)
            spectrum_df = spectrum_df.groupby('interval')['intensity'].sum()
            spectrum_df = spectrum_df.reset_index()
            convert_dict = {'interval': int} 
            spectrum_df = spectrum_df.astype(convert_dict) 
            spectrum_dict[key] = spectrum_df
        total_spectrum_dict[j] = spectrum_dict
        j+=1

#decide the final set of segmentation intervals for all readings
maximal_interval = []
minimal_interval = []
for key_1 in total_spectrum_dict.keys():
    spectrum_dict = total_spectrum_dict[key_1]
    for key_2 in spectrum_dict.keys():
        maximal_interval.append(max(spectrum_dict[key_2].interval))
        minimal_interval.append(min(spectrum_dict[key_2].interval))
interval_upper_bound = int(max(maximal_interval))
interval_lower_bound = int(min(minimal_interval))
spectrum_range_feature_set = set(range(interval_lower_bound,interval_upper_bound+1,1)) #this set contains the finalized feature intervals

#transform all readings into standard data frame with the same features(spectrum intervals)
for key_1 in total_spectrum_dict.keys():
    spectrum_dict = total_spectrum_dict[key_1]
    for key_2 in spectrum_dict.keys():
        spectrum_df = spectrum_dict[key_2]
        add_interval_set = spectrum_range_feature_set-set(spectrum_df.interval)
        complemented_spectrum_df = spectrum_df
        for interval in add_interval_set:
            complemented_spectrum_df = complemented_spectrum_df.append({'interval':int(interval),'intensity':float(0)}, ignore_index=True).sort_values(by=['interval'])
        spectrum_dict[key_2] = complemented_spectrum_df 


# In[13]:


#check if all data frames have the same number of features(spectrum intervals)
cnt = 0
abnormal_case = {}
for key_1 in total_spectrum_dict.keys():
    spectrum_dict = total_spectrum_dict[key_1]
    for key_2 in spectrum_dict.keys():
        spectrum_df = spectrum_dict[key_2]
        if spectrum_df.shape[0]!=len(spectrum_range_feature_set)or spectrum_df.shape[1]!=2:
            abnormal_case[key_2] = spectrum_df
            cnt += 1
print('number of incorrect data frame is:',cnt)


# In[101]:


spectrum_final_df_dict = {}
for key_1 in total_spectrum_dict.keys():
    spectrum_dict = total_spectrum_dict[key_1]
    spectrum_final_df = pd.DataFrame([])
    for key_2 in spectrum_dict.keys():
        spectrum_df_t = spectrum_dict[key_2].T
        if spectrum_df_t.shape[1] != 0:
            spectrum_df_t = spectrum_df_t.reset_index(drop=True).iloc[[1]]
            spectrum_final_df = spectrum_final_df.append(spectrum_df_t)
        else:
            continue
    spectrum_final_df_dict[key_1]=spectrum_final_df
# each element in spectrum_final_df_dict is a data frame corresponding to all readings in a single mzML file
# each data frame has 20 columns corresponding to 20 intervals for spectrum as features of each reading

