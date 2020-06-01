import numpy as np
import lxml as lx
from pyteomics import mzxml, pylab_aux
from joblib import Parallel, delayed
import multiprocessing
import argparse
import os
import copy
import time
import csv

outputs = []
file_obj = {}
num_cores = multiprocessing.cpu_count()

# import mzXML files to read in
# filepath: path to directory containing mzXML files to be read in
# populates outputs: list of pyteomics objects
# populates file_obj: dict linking pyteomics objects (key) to file names(value)
def import_files(filepath):
    os.chdir(filepath)
    directory = os.listdir(os.getcwd())
    global outputs
    global file_obj
    i = 0

    for file in directory:
        if file.endswith("mzXML"):
            obj = mzxml.read(file)
            file_obj[obj] = file
            outputs.append(obj)

# convert outputs into list of tuples(time, signal intensity)
# obj: pyteomics object to be converted to time, intensity
# returns time_inten: 2D array of retention time values matched with intensity
def parse_outputs(obj):
    time = []
    inten = []
    time_inten = []

    for signal in obj:
        time.append((signal['retentionTime']))
        inten.append(np.sum(signal['intensity array']))
    time = np.array(time)
    inten = np.array(inten)
    time_inten = np.vstack((time, inten))
    time_inten = time_inten.transpose()

    title, end = file_obj[obj].split(".")
    np.savetxt(title + ".csv", time_inten, delimiter = ',', fmt = '%.8f')
    print(time_inten.shape)
    return time_inten

parser = argparse.ArgumentParser(description = 'Convert mzXML files to csv')
parser.add_argument('-i', type = str, help = 'path to folder containing mzXML files')
# parser.add_argument('output folder', metavar = 'o', type = str, help = 'path to folder to put csv files into')
args = parser.parse_args()
print(args.i)

input_folder = args.i

if input_folder is None:
    input_folder = os.getcwd()

import_files(input_folder)
time_inten = Parallel(n_jobs = num_cores)(delayed(parse_outputs)(x) for x in outputs)
