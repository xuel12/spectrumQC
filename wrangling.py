#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 20:59:16 2020

@author: xuel12
"""

import re
import sys
import os
import pickle

try: 
    os.chdir('/Users/xuel12/Documents/MSdatascience/DS5500datavis/project1/spectrumQC/')
    print("Current directory is {}".format(os.getcwd()))
except: 
    print("Something wrong with specified directory. Exception- ", sys.exc_info())

import numpy as np
import pandas as pd
from pyteomics import mzml

import constants


def mzML2dict(data_dir, temp_dir, bin_size):
    os.chdir(temp_dir)
    # j = 0
    exist_spectrum_dict = {}
    # filename = mzML_file_names[2]
    for filename in os.listdir(data_dir): #iterate through all mzML file
        speclist = []
        if (re.search('\\.mzML$', filename)):
            print('parsing file ', filename)
            with mzml.read(os.path.join(data_dir, filename), 'r') as reader:
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
                exist_spectrum_dict[filename] = spectrum_df.astype(convert_dict)
                # j+=1
                
    print('Combining files ...')
    interval_upper_bound = 2000//bin_size
    interval_lower_bound = 0
    spectrum_range_feature_set = set(range(interval_lower_bound,interval_upper_bound,1)) #this set contains the finalized feature intervals
    
    #transform all readings into standard data frame with the same features(spectrum intervals)
    complemented_spectrum_dict = {}
    total_spectrum_dict = {}
    # key_1 = mzML_file_names[2]
    for key_1 in exist_spectrum_dict.keys():
        spectrum_dict = exist_spectrum_dict[key_1]
        complemented_spectrum_list = []
        for key_2 in set(spectrum_dict.specid):
            spectrum_df = spectrum_dict[spectrum_dict.specid.eq(key_2)]
            add_interval_set = spectrum_range_feature_set-set(spectrum_df.interval)
            complemented_spectrum_list.append(np.column_stack(([key_2]*len(add_interval_set), list(add_interval_set), [float(0)]*len(add_interval_set))))
        complemented_spectrum_np = np.concatenate(complemented_spectrum_list, axis=0)
        spectrum_df = pd.DataFrame({'specid':complemented_spectrum_np[:,0], 'interval':complemented_spectrum_np[:,1],'intensity':complemented_spectrum_np[:,2]})            
        convert_dict = {'specid': int, 'interval': int} 
        complemented_spectrum_dict[key_1] = spectrum_df.astype(convert_dict)    
        total_spectrum_dict[key_1] = pd.concat([exist_spectrum_dict[key_1], complemented_spectrum_dict[key_1]]).sort_values(by=['specid','interval'])
    
    print('Checking combined spectra ...')
    for key_1 in total_spectrum_dict.keys():
        spectrum_dict = total_spectrum_dict[key_1]
        count_df = spectrum_dict[['specid','interval']].groupby(['specid']).count()        
        if (len(count_df[count_df.interval != len(spectrum_range_feature_set)]) > 0):
            print('there is incorrect data frame in ', key_1)
            errorlist = count_df.index[count_df['interval'] != len(spectrum_range_feature_set)].tolist()
            spectrum_dict = spectrum_dict[~spectrum_dict['specid'].isin(errorlist)]
        total_spectrum_dict[key_1] = spectrum_dict
        
    print('Saving spectra into file ...')
    f = open("total_spectrum_dict.pkl","wb")
    pickle.dump(total_spectrum_dict,f)
    f.close()
    print('Spectra file saving DONE!')
    
    return 1

def evidenceDF(data_dir, temp_dir):
    os.chdir(temp_dir)
    evidence = pd.read_csv(data_dir + '/evidence.txt', sep='\t')
    evidence = evidence.rename(columns={"Raw file": "mzML_name"})
    
    f = open("evidence_df.pkl", "wb")
    pickle.dump(evidence,f)
    f.close()
    
def mzMLfilename(data_dir):
    mzML_file_names = []
    for filename in os.listdir(data_dir):
        if (re.search('\\.mzML$', filename)):
            mzML_file_names.append(filename)
    return mzML_file_names

if __name__ == "__main__":
    
    os.chdir(constants.CODE_DIR)
    data_dir = constants.DATA_DIR
    temp_dir = constants.TEMP_DIR
    bin_size = constants.BIN_SIZE
    
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    mzML2dict(data_dir, temp_dir, bin_size)
    evidenceDF(data_dir, temp_dir)
    
