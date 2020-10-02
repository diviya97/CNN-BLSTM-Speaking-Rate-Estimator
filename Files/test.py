#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Importing Packages

import keras
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from keras.layers import multiply
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation, TimeDistributed, Bidirectional
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Flatten
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from scipy.stats import pearsonr
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from keras.models import load_model

timit_mu = [4.838954, 4.834209, 4.838362, 4.832054, 4.836015]
timit_sigma = [0.817343, 0.808527, 0.803453, 0.803760, 0.807249]
swbd_mu = [4.722949, 4.724373, 4.739988, 4.743317, 4.746103]
swbd_sigma = [1.471280, 1.434821, 1.431199, 1.470063, 1.484794]
timit_swbd_mu = [4.7766869860637255, 4.775252952230831, 4.785558354333942, 4.784423139151849, 4.787753645100162] 
timit_swbd_sigma = [1.2143831573688817, 1.1877997366214124, 1.1836024232480609, 1.2088076079970034, 1.2195228607776702]

PATH = '/Data/data_timit'

EPSILON = 0.0000001
FEATURES = 20

#Converting sylNucleiLocs to Speaking Rate
#Speaking Rate(Y_labels) = (No.of Syllable Nuclei)/(No. of Points in Ftr1)
def speaking_Rate(Y):
    num_in_fold = len(Y[0])
    Y_labels = np.zeros((5,num_in_fold), dtype='float64')

    for i in range(5):
        for j in range(num_in_fold):
            y = Y[i][j]
            a = np.divide(np.count_nonzero(y == 1), y.shape[1], dtype='float64')
            Y_labels[i][j] = a*100     #Multiplying by 100 as there are 100 frames per sec 
            
    return Y_labels


# In[24]:


# normalizing given array between -1 to 1 
def normalize_X(x, axis):
    mu = np.mean( x, axis=axis, dtype='float64', keepdims=True)
    sigma = ( np.std( x, axis=axis, dtype='float64', keepdims=True) + EPSILON)
    #print('mu_shape: ', mu.shape)                                   #(1,20)
    normalized_x = (x-mu)/sigma
    x = normalized_x/10
    return x

def normalize_Y(y_train, y_val, y_test):
    
    mu_tr = np.mean( y_train, dtype='float64')
    sigma_tr = ( np.std( y_train, dtype='float64') + EPSILON)
    
    y_train = (y_train-mu_tr)/(sigma_tr*4)
    y_val = (y_val-mu_tr)/(sigma_tr*4)
    y_test = (y_test-mu_tr)/(sigma_tr*4)
    
    return y_train, y_val, y_test, mu_tr, sigma_tr


def denormalize_Y(y, mu_tr, sigma_tr):
    y = (y*sigma_tr*4) + mu_tr
    y = np.squeeze(y)
    return y

def load_test_data():

    print('Loading Data.....')

    # Loading indices of files from SwitchboardFolds.mat in folds_ind
    LOC = PATH
    file_to_open = '/TIMITFolds.mat'
    folds_file = scipy.io.loadmat(LOC + file_to_open)
    folds = folds_file['foldInds']
    folds = np.reshape(folds, (5, -1))
    num_in_fold = folds[0][0].shape[0]
    folds_ind = np.zeros((5, num_in_fold), dtype=int)
    for i in range(5):
        folds_ind[i] = folds[i][0].flatten()
    # Loading file names from timitSpurtNames
    LOC = PATH
    file_to_open = '/timitSpurtNames.mat'
    file = scipy.io.loadmat(LOC+file_to_open)
    file_names = file['timitSpurtName']

    # Declaring Variables to store the data
    X_Total = []
    Y_Total = []

    # Loading Data
    parent = PATH+'/data'
    suffix = '.mat'

    for i in range(5):
        x = []
        y = []
        for j in range(num_in_fold):
            index = folds_ind[i][j]-1
            filename = file_names[index][0][0]
            path = os.path.join(parent, filename+suffix)
            data = scipy.io.loadmat(path)
            a = data['Ftr2']
            b = np.squeeze(data['pch_interp'])
            #appending 'pch_interp' to 'Ftr2'
            x_temp = np.zeros((a.shape[0],20))
            x_temp[:,:-1] = a
            x_temp[:,-1] = b
            x.append(normalize_X(x_temp,0))
            y.append(data['sylNucleiLocs'])
        X_Total.append(x)  # shape 5 X 1260 X seq_length X 20
        Y_Total.append(y)  # shape 5 X 1260 X 1 X seq_length
        
    #Converting ground truth to speaking rate    
    Y_Total = speaking_Rate(Y_Total)
    
    return X_Total, Y_Total

# In[28]:


#Main

X, Y = load_test_data()
test_pearson_coeff = []

for fold in range(5):

    print('fold : '+str(fold+1))
    val_index = (fold+1)%5
    
    fName = '/SavedModels/timit/train_timit_fold_'+ str(fold+1)
    model = load_model(fName+'_.h5')
    
    #Predicting values for test data
    num_test = len(X[fold])
    x_test = X[fold]
    print('Test on '+str(num_test)+' samples ')
    
    test_pred = np.zeros(num_test)
    for j in range(num_test):
        xt = np.resize(x_test[j],(1, x_test[j].shape[0], FEATURES))
        test_pred[j] = model.predict(xt)
    
    #Denormalizing Y
    test_pred = denormalize_Y(test_pred, timit_mu[fold], timit_sigma[fold])
    
    print('Calculating Pearson Coefficient.....')
    test_corr, _ = pearsonr(np.squeeze(test_pred), np.squeeze(Y[fold]))
    test_pearson_coeff.append(test_corr)
        
    print('True Test Values : ' + str(Y[fold]))
    print('Test Predictions : ' + str(test_pred))
    print('Test Pearson Coefficient : ' + str(test_pearson_coeff))

test_pearson_coeff = np.array(test_pearson_coeff)
test_pearson_coeff_avg = np.sum(test_pearson_coeff)/5
print('Test Pearson Coefficient Avg: '+str(test_pearson_coeff_avg))

# In[ ]:


