#!/usr/bin/env python
# coding: utf-8

# In[146]:


import pyteomics
import sys
import os
import numpy as np
import pandas as pd
from pyteomics import tandem
import pymzml
import argparse

try: 
    # os.chdir('F://')
    os.chdir('/Users/xuel12/Documents/MSdatascience/DS5500datavis/project1')
    print("Current directory is {}".format(os.getcwd()))
except: 
    print("Something wrong with specified directory. Exception- ", sys.exc_info())
    



# In[154]:


from pyteomics import mzml, auxiliary
with mzml.read('data/01625b_GB2-TUM_first_pool_10_01_01-DDA-1h-R2.mzML') as reader:
    auxiliary.print_tree(next(reader))
#     reader['scanList']


# In[153]:


reader = mzml.read('data/01625b_GB2-TUM_first_pool_10_01_01-DDA-1h-R2.mzML')
mzlist = {}
intensitylist = {}
i = 0
for spec in reader:
    if (spec['ms level'] == 2):
        mzlist[i] = spec['m/z array'].tolist()
        intensitylist[i] = spec['intensity array'].tolist()
    i += 1

   

# reader.binary_array_record.decode

# print(next(reader))
# # dir(reader)
# # print(reader.schema_info)
# reader_schema = reader.schema_info
# reader_schema.keys()
# reader_schema['ints']

