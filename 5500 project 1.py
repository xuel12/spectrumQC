#!/usr/bin/env python
# coding: utf-8

# In[146]:


import pyteomics
import os
import numpy as np
import pandas as pd
from pyteomics import tandem
import pymzml
import argparse
os.chdir('F://')


# In[154]:


from pyteomics import mzml, auxiliary
with mzml.read('01625b_GA1-TUM_first_pool_1_01_01-DDA-1h-R2.mzML') as reader:
    auxiliary.print_tree(next(reader))
#     reader['scanList']


# In[153]:


reader = mzml.read('01625b_GA1-TUM_first_pool_1_01_01-DDA-1h-R2.mzML') 
reader.binary_array_record.decode
# print(next(reader))
# # dir(reader)
# # print(reader.schema_info)
# reader_schema = reader.schema_info
# reader_schema.keys()
# reader_schema['ints']

