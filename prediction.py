#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 21:44:34 2020

@author: xuel12
"""


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
from sklearn.metrics import roc_auc_score

import constants
import wrangling


def predictDataset(predict_dir, bin_size, predictfile_names):
    os.chdir(predict_dir)
    # # open evidence pickled data
    # f = open(temp_dir + 'evidence_df.pkl', 'rb')
    # evidence = pickle.load(f)
    # f.close()
    
    # open spectra pickled data
    f = open(predict_dir + 'total_spectrum_dict.pkl', 'rb')
    total_spectrum_dict = pickle.load(f)
    f.close()
    
    bin_num = 2000//bin_size
    #combine all mzML file together for training and testing set split
    totoal_spectrum_df = pd.DataFrame()
    
    rename_column = ['spectrum_id']+[str(i*bin_size)+'-'+str(i*bin_size+bin_size) for i in range(bin_num)]+['label']
    # rename_column = ['spectrum_id']+[str(i*10) for i in range(200)]+['label']
    file_name = predictfile_names[0]
    for file_name in predictfile_names:
        # positive_label = evidence[evidence['mzML_name']==str(file_name.split('.')[0])]
        # positive_label = positive_label['MS/MS scan number'].to_list()
        spec_df = total_spectrum_dict[file_name]
        spec_df_ready = spec_df.pivot(index='specid',columns='interval',values='intensity')
        spec_df_ready = spec_df_ready.reset_index()
        # spec_df_ready['label'] =spec_df_ready['specid'].isin(positive_label)
        spec_df_ready['label'] = -1

        totoal_spectrum_df = pd.concat([totoal_spectrum_df, spec_df_ready])
    totoal_spectrum_df.columns = rename_column
    
    # random.seed(100)
    # train_dict = {}
    # test_dict = {}
    # train_dict['X_train'],test_dict['X_test'], train_dict['y_train'], test_dict['y_test'] = train_test_split(totoal_spectrum_df.iloc[:, 1:(bin_num+1)], totoal_spectrum_df['label'], test_size=0.2, random_state=42)
    predict_dict = {}
    predict_dict['spectrum_id'] = totoal_spectrum_df['spectrum_id']
    predict_dict['X_predict'] = totoal_spectrum_df.iloc[:, 1:(bin_num+1)]
    predict_dict['y_predict'] = totoal_spectrum_df['label']

    f = open(predict_dir + "prediction_dataset.pkl","wb")
    pickle.dump(predict_dict, f)
    f.close()
    
    return 1

#define predict function
def predict_spectrum_quality(predict_dir, model_dir, out_dir):
    # open predict pickled data
    f = open(predict_dir + 'prediction_dataset.pkl', 'rb')
    predict_dict = pickle.load(f)
    f.close()
    
    X_predict = predict_dict['X_predict']
    y_predict = np.random.randint(2, size=len(predict_dict['X_predict']))
    
    # load trained model
    f = open(model_dir + 'training_model.pkl', 'rb')
    clf = pickle.load(f)
    f.close()
        

    prediction_result = {}
    prediction_result['model_params'] = clf.get_params()
    prediction_result['AUC'] = roc_auc_score(y_predict, clf.predict_proba(X_predict)[:, 1])
    prediction_result['predict'] = pd.concat([predict_dict['spectrum_id'], pd.Series(clf.predict(X_predict))], axis=1,
              names=['spectrum_id', 'predict_label'])
    
    # save prediction result
    print('prediction result is returned')
    f = open(predict_dir + "prediction_result.pkl","wb")
    pickle.dump(prediction_result, f)
    f.close()
    
    prediction_result['predict'].to_csv(out_dir + 'predict_result.csv', index=False)
    
    return(prediction_result)


if __name__ == "__main__":
    
    os.chdir(constants.CODE_DIR)
    data_dir = constants.DATA_DIR
    temp_dir = constants.TEMP_DIR
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    model_dir = constants.MODEL_DIR
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    out_dir = constants.OUT_DIR
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    predict_dir = constants.PREDICT_DIR

    bin_size = constants.BIN_SIZE

    # prepare prediction set
    predictfile_names = wrangling.mzMLfilename(predict_dir)
    wrangling.mzML2dict(predict_dir, predict_dir, bin_size)
    predictDataset(predict_dir, bin_size, predictfile_names)

    # apply trained model for prediction
    prediction_result = predict_spectrum_quality(predict_dir, model_dir, out_dir)
    