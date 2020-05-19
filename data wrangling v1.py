#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import re
import numpy as np
import pandas as pd
import pyteomics
from pyteomics import tandem
# import pymzml
import argparse
from pyteomics import mzml, auxiliary
from matplotlib import pyplot as plt


# In[3]:


try: 
    # os.chdir('F://DS5500_project_1')
    os.chdir('/Users/xuel12/Documents/MSdatascience/DS5500datavis/project1')
    print("Current directory is {}".format(os.getcwd()))
except: 
    print("Something wrong with specified directory. Exception- ", sys.exc_info())


# In[5]:

datadir = os.getcwd()+'/data'
j = 0
exist_spectrum_dict = {}
speclist = []
bin_size = 10
for filename in os.listdir(datadir): #iterate through all mzML file
    if (re.search('\\.mzML$', filename)):
        with mzml.read(os.path.join(datadir, filename), 'r') as reader:
            for spec in reader:
                if (spec['ms level'] == 2):
                    tmp_id = [int(spec['id'].split('=')[3])]*len(spec['m/z array'])
                    speclist.append(np.column_stack((tmp_id, spec['m/z array'],spec['intensity array'])))
            
            spectrum_np = np.concatenate(speclist, axis=0)
            spectrum_df = pd.DataFrame({'specid':spectrum_np[:,0], 'location':spectrum_np[:,1],'intensity':spectrum_np[:,2]})            
            spectrum_df = spectrum_df.assign(interval=spectrum_df['location']//bin_size)
            spectrum_df = spectrum_df.groupby(['specid','interval'])['intensity'].sum()
            spectrum_df = spectrum_df.reset_index()
            convert_dict = {'specid': int, 'interval': int} 
            exist_spectrum_dict[j] = spectrum_df.astype(convert_dict)
                
            # spectrum_dict = {}
            # for key in intensitydict.keys(): #create the raw data frame each of which represents a spectrum reading
            #     spectrum_dict[key] = pd.DataFrame({'location':mzdict[key],'intensity':intensitydict[key]})
            # for key in spectrum_dict.keys(): #segment spectrums by intervals with length of 100
            #     spectrum_df = spectrum_dict[key]
            #     spectrum_df = spectrum_df.assign(interval=spectrum_df['location']//100+1)
            #     spectrum_df = spectrum_df.groupby('interval')['intensity'].sum()
            #     spectrum_df = spectrum_df.reset_index()
            #     convert_dict = {'interval': int} 
            #     spectrum_df = spectrum_df.astype(convert_dict) 
            #     spectrum_dict[key] = spectrum_df
            # total_spectrum_dict[j] = spectrum_dict
            j+=1

#decide the final set of segmentation intervals for all readings
# maximal_interval = []
# minimal_interval = []
# key_1 = 0
# for key_1 in total_spectrum_dict.keys():
#     spectrum_dict = total_spectrum_dict[key_1]
#     for key_2 in spectrum_dict.keys():
#         maximal_interval.append(max(spectrum_dict[key_2].interval))
#         minimal_interval.append(min(spectrum_dict[key_2].interval))
# interval_upper_bound = int(max(maximal_interval))
# interval_lower_bound = int(min(minimal_interval))
interval_upper_bound = 2000//bin_size
interval_lower_bound = 0
spectrum_range_feature_set = set(range(interval_lower_bound,interval_upper_bound,1)) #this set contains the finalized feature intervals

#transform all readings into standard data frame with the same features(spectrum intervals)
complemented_spectrum_dict = {}
complemented_spectrum_list = []
total_spectrum_dict = {}
for key_1 in exist_spectrum_dict.keys():
    spectrum_dict = exist_spectrum_dict[key_1]
    for key_2 in set(spectrum_dict.specid):
        spectrum_df = spectrum_dict[spectrum_dict.specid.eq(key_2)]
        add_interval_set = spectrum_range_feature_set-set(spectrum_df.interval)
        complemented_spectrum_list.append(np.column_stack(([key_2]*len(add_interval_set), list(add_interval_set), [float(0)]*len(add_interval_set))))
    complemented_spectrum_np = np.concatenate(complemented_spectrum_list, axis=0)
    spectrum_df = pd.DataFrame({'specid':complemented_spectrum_np[:,0], 'interval':complemented_spectrum_np[:,1],'intensity':complemented_spectrum_np[:,2]})            
    convert_dict = {'specid': int, 'interval': int} 
    complemented_spectrum_dict[key_1] = spectrum_df.astype(convert_dict)    

    total_spectrum_dict[key_1] = pd.concat([exist_spectrum_dict[key_1], complemented_spectrum_dict[key_1]]).sort_values(by=['specid','interval'])

# for key_1 in total_spectrum_dict.keys():
#     spectrum_dict = total_spectrum_dict[key_1]
#     for key_2 in spectrum_dict.keys():
#         spectrum_df = spectrum_dict[key_2]
#         add_interval_set = spectrum_range_feature_set-set(spectrum_df.interval)
#         complemented_spectrum_df = spectrum_df
#         for interval in add_interval_set:
#             complemented_spectrum_df = complemented_spectrum_df.append({'interval':int(interval),'intensity':float(0)}, ignore_index=True).sort_values(by=['interval'])
#         spectrum_dict[key_2] = complemented_spectrum_df 


# In[13]:


#check if all data frames have the same number of features(spectrum intervals)
for key_1 in total_spectrum_dict.keys():
    spectrum_dict = total_spectrum_dict[key_1]
    count_df = spectrum_dict[['specid','interval']].groupby(['specid']).count()
    if (len(count_df[count_df.interval != len(spectrum_range_feature_set)]) > 0):
       print('there is incorrect data frame in ', key_1)

# cnt = 0
# abnormal_case = {}
# for key_1 in total_spectrum_dict.keys():
#     spectrum_dict = total_spectrum_dict[key_1]
#     for key_2 in spectrum_dict.keys():
#         spectrum_df = spectrum_dict[key_2]
#         if spectrum_df.shape[0]!=len(spectrum_range_feature_set)or spectrum_df.shape[1]!=2:
#             abnormal_case[key_2] = spectrum_df
#             cnt += 1
# print('number of incorrect data frame is:',cnt)


# In[101]:


# spectrum_final_df_dict = {}
# for key_1 in total_spectrum_dict.keys():
#     spectrum_dict = total_spectrum_dict[key_1]
#     spectrum_final_df = pd.DataFrame([])
#     for key_2 in spectrum_dict.keys():
#         spectrum_df_t = spectrum_dict[key_2].T
#         if spectrum_df_t.shape[1] != 0:
#             spectrum_df_t = spectrum_df_t.reset_index(drop=True).iloc[[1]]
#             spectrum_final_df = spectrum_final_df.append(spectrum_df_t)
#         else:
#             continue
#     spectrum_final_df_dict[key_1]=spectrum_final_df
    
# each element in spectrum_final_df_dict is a data frame corresponding to all readings in a single mzML file
# each data frame has 20 columns corresponding to 20 intervals for spectrum as features of each reading

