#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 21:44:34 2020

@author: xuel12
"""

import sys
import os
import random
import pickle

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  # doctest: +SKIP

try: 
    os.chdir('/Users/xuel12/Documents/MSdatascience/DS5500datavis/project1/spectrumQC/')
    print("Current directory is {}".format(os.getcwd()))
except: 
    print("Something wrong with specified directory. Exception- ", sys.exc_info())

import constants
import wrangling


def trainingDataset(temp_dir, bin_size, mzML_file_names):
    # os.chdir(temp_dir)
    # open evidence pickled data
    f = open(temp_dir + 'evidence_df.pkl', 'rb')
    evidence = pickle.load(f)
    f.close()
    
    # open spectra pickled data
    f = open(temp_dir + 'total_spectrum_dict.pkl', 'rb')
    total_spectrum_dict = pickle.load(f)
    f.close()
    
    bin_num = 2000//bin_size
    #combine all mzML file together for training and testing set split
    totoal_spectrum_df = pd.DataFrame()
    
    rename_column = ['spectrum_id']+[str(i*bin_size)+'-'+str(i*bin_size+bin_size) for i in range(bin_num)]+['label']
    # rename_column = ['spectrum_id']+[str(i*10) for i in range(200)]+['label']
    # file_name = mzML_file_names[3]
    # file_name = mzML_file_names[1]
    for file_name in mzML_file_names:
        positive_label = evidence[evidence['mzML_name']==str(file_name.split('.')[0])]
        positive_label = positive_label['MS/MS scan number'].to_list()
        spec_df = total_spectrum_dict[file_name].copy()
        # tmp = spec_df.duplicated('specid')
        spec_df_ready = spec_df.pivot(index='specid',columns='interval',values='intensity')
        spec_df_ready = spec_df_ready.reset_index()
        spec_df_ready['label'] =spec_df_ready['specid'].isin(positive_label)
        # spec_df_ready.loc[spec_df_ready['specid']==7662]
        totoal_spectrum_df = pd.concat([totoal_spectrum_df, spec_df_ready], ignore_index=True)
    totoal_spectrum_df.columns = rename_column

    random.seed(100)
    train_dict = {}
    test_dict = {}
    train_dict['X_train'],test_dict['X_test'], train_dict['y_train'], test_dict['y_test'] = train_test_split(totoal_spectrum_df.iloc[:, 1:(bin_num+1)], totoal_spectrum_df['label'], test_size=0.2, random_state=42)
    
    f = open(temp_dir + "trainset.pkl","wb")
    pickle.dump(train_dict, f)
    f.close()
    
    f = open(temp_dir + "testset.pkl","wb")
    pickle.dump(test_dict, f)
    f.close()
    
    return 1

#define modelling function
def modelling_spectrum_quality(temp_dir, model_dir, method, param_grid):
    # open training pickled data
    f = open(temp_dir + 'trainset.pkl', 'rb')
    train_dict = pickle.load(f)
    f.close()
    # open testing pickled data
    f = open(temp_dir + 'testset.pkl', 'rb')
    test_dict = pickle.load(f)
    f.close()
    
    upperms = 150
    X_train = train_dict['X_train'].iloc[:,:upperms]
    y_train = train_dict['y_train']
    X_test = test_dict['X_test'].iloc[:,:upperms]
    y_test = test_dict['y_test']

    random.seed(200)
    #Generate random number due to slowness of svm
    randomlist_train = random.sample(range(len(X_train)), 10000)    

    if method == 'rf' or method == 'both':
        print('Random forest training starts ...')
        #random forest modelling:
        clf = RandomForestClassifier()
        param_grid_rf = param_grid['rf']
        # param_grid = { "min_samples_leaf" : [2, 5, 10], "min_samples_split" : [5, 10, 25], "n_estimators": [50, 100, 200]}
        gs = GridSearchCV(estimator=clf, param_grid=param_grid_rf, scoring='f1', cv=3, n_jobs=-1) 
        gs = gs.fit(X_train.iloc[randomlist_train], y_train.iloc[randomlist_train])
    
        clf_opt = gs.best_estimator_
        clf_opt.fit(X_train.iloc[randomlist_train], (y_train.iloc[randomlist_train] == True))
        # feature_scores = pd.Series(clf_opt.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        feature_scores_df = pd.Series(clf_opt.feature_importances_, index=X_train.columns).sort_values(ascending=False).reset_index()
        auc_score = roc_auc_score(y_test, clf_opt.predict_proba(X_test)[:, 1])
        
        print('random forest model:')
        print('parameters of optimal model are:',gs.best_params_)
        # print('top 10% important features are:',feature_scores_df.head(round(feature_scores_df.shape[0]*0.1)))
        print('AUC score of optimal model is:',auc_score)
    
        prediction_model = {}
        prediction_model['model_params'] = clf_opt.get_params()
        prediction_model['AUC'] = roc_auc_score(y_test, clf_opt.predict_proba(X_test)[:, 1])
        prediction_model['top_important_features'] = feature_scores_df.head(round(feature_scores_df.shape[0]*0.1))

    if method == 'svm' or method == 'both':
        print('SVM training starts ...')
        scaler = StandardScaler()  # doctest: +SKIP
        # Don't cheat - fit only on training data
        scaler.fit(X_train)  # doctest: +SKIP
        X_train_np = scaler.transform(X_train)  # doctest: +SKIP
        # apply same transformation to test data
        X_test_np = scaler.transform(X_test)  # doctest: +SKIP
    
        #SVM modelling:
        svc = SVC(probability=True)
        param_grid_svc = param_grid['svm']
        # param_grid_svc = { "kernel" : ['linear', 'rbf']}
        gs_svc = GridSearchCV(estimator=svc, param_grid=param_grid_svc, scoring='f1', cv=3, n_jobs=-1) 
        gs_svc = gs_svc.fit(X_train_np[randomlist_train,:], (y_train.iloc[randomlist_train] == True).astype(int))
    
        clf_opt_svc = gs_svc.best_estimator_
        clf_opt_svc.fit(X_train_np[randomlist_train,:],(y_train.iloc[randomlist_train] == True).astype(int))
        print('support vector machine:')
        print('parameters of optimal model are:',gs_svc.best_params_)
        print('AUC score of optimal model is:',roc_auc_score(y_test, clf_opt_svc.predict_proba(X_test_np)[:, 1]))
    
    if method == 'mlp':
        scaler = StandardScaler()  # doctest: +SKIP
        # Don't cheat - fit only on training data
        scaler.fit(X_train)  # doctest: +SKIP
        X_train_np = scaler.transform(X_train)  # doctest: +SKIP
        # apply same transformation to test data
        X_test_np = scaler.transform(X_test)  # doctest: +SKIP
        
        #SVM modelling:
        mlp = MLPClassifier()
        param_grid_mlp = param_grid['mlp']
        gs_mlp = GridSearchCV(estimator=mlp, param_grid=param_grid_mlp, scoring='f1', cv=3, n_jobs=-1) 
        gs_mlp = gs_mlp.fit(X_train.iloc[randomlist_train], (y_train.iloc[randomlist_train] == True).astype(int))

        clf_opt_mlp = gs_mlp.best_estimator_
        clf_opt_mlp.fit(X_train.iloc[randomlist_train],(y_train.iloc[randomlist_train] == True).astype(int))
        print('MLP:')
        print('parameters of optimal model are:',gs_mlp.best_params_)
        print('AUC score of optimal model is:',roc_auc_score(y_test, clf_opt_mlp.predict_proba(X_test)[:, 1]))

    print('Training is DONE!')

    if method == 'rf':
        final_model = clf_opt
    elif method == 'svm':
        final_model = clf_opt_svc
    elif method == 'mlp':
        final_model = clf_opt_mlp
    else:
        #select the better model from random forest and SVM by AUC score:
        if roc_auc_score(y_test, clf_opt_svc.predict_proba(X_test)[:, 1])> roc_auc_score(y_test, clf_opt.predict_proba(X_test)[:, 1]):
            print('best model for is SVM with AUC:',roc_auc_score(y_test, clf_opt_svc.predict_proba(X_test)[:, 1]))
            print('model parameter is:',gs_svc.best_params_)
            final_model = clf_opt_svc
        else: 
            print('best model for is random forest with AUC:',roc_auc_score(y_test, clf_opt.predict_proba(X_test)[:, 1]))
            print('model parameter is:',gs.best_params_)
            final_model = clf_opt
            
    print('selected model is returned')
    
    f = open(model_dir + "training_model.pkl","wb")
    pickle.dump(final_model, f)
    f.close()
    
    final_result = {}
    final_result['X_train'] = X_train
    final_result['y_train'] = y_train
    final_result['X_test'] = X_test
    final_result['y_test'] = y_test
    final_result['model'] = final_model
    
    return final_result


if __name__ == "__main__":
    
    os.chdir(constants.CODE_DIR)
    data_dir = constants.DATA_DIR
    temp_dir = constants.TEMP_DIR
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    model_dir = constants.MODEL_DIR
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    bin_size = constants.BIN_SIZE
    

    # prepare training set
    mzML_file_names = wrangling.mzMLfilename(data_dir)
    # parse mzML files to dictionary
    wrangling.mzML2dict(data_dir, temp_dir, bin_size)
    wrangling.evidenceDF(data_dir, temp_dir)
    trainingDataset(temp_dir, bin_size, mzML_file_names)
    
    param_grid = {'rf': {"min_samples_leaf" : [2], "min_samples_split" : [5], "n_estimators": [50]},\
                  'svm': {"kernel" : ['rbf']},
                  'mlp': {"activation" : ['relu']}}
    model = modelling_spectrum_quality(temp_dir, model_dir, method='rf', param_grid=param_grid)
    