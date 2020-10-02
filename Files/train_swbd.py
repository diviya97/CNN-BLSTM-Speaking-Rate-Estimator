#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Importing Packages

import keras
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
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


EPOCHS_WITHOUT_VAL= 12
EPSILON = 0.0000001
FEATURES = 20
PATH = '/Data/data_swbd'
EPOCHS = 25

pooling_size = 2
num_filters = 64
filter_length = 5

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


def split_data(X, Y, fold, val_index):

    x_test = X[fold]
    y_test = Y[fold]

    x_val = X[val_index]
    y_val = Y[val_index]

    temp_x_train = []
    temp_y_train = []

    for k in range(5):
        if k != fold and k != val_index:
            temp_x_train.append(X[k])
            temp_y_train.append(Y[k])

    #appending examples of different folds together
    x_train = [item for sublist in temp_x_train for item in sublist]
    y_train = [item for sublist in temp_y_train for item in sublist]

    return x_train, y_train, x_val, y_val, x_test, y_test


def load_data():

    print('Loading Data.....')

    # Loading indices of files from SwitchboardFolds.mat in folds_ind
    LOC = PATH
    file_to_open = '/SwitchboardFolds.mat'
    folds_file = scipy.io.loadmat(LOC + file_to_open)
    folds = folds_file['foldInds']
    folds = np.reshape(folds, (5, -1))
    num_in_fold = folds[0][0].shape[0]
    folds_ind = np.zeros((5, num_in_fold), dtype=int)
    for i in range(5):
        folds_ind[i] = folds[i][0].flatten()
    # Loading file names from estSpurtNames
    LOC = PATH
    file_to_open = '/estSpurtNames.mat'
    file1 = scipy.io.loadmat(LOC+file_to_open)
    file_names = file1['estSpurtName']

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
	    # 'Ftr2' id 19Sbes
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


# In[27]:

 # Model
def build_model(num_filters, filter_length, pooling_size=2):
    model = Sequential()

    model.add(Conv1D(filters=num_filters,
                     kernel_size=filter_length,
                     strides=1,
                     batch_input_shape=(None, None, FEATURES)))  # (batch, steps, channels)
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=pooling_size)),
    model.add(Dropout(0.20))

    model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)))
    model.add(Bidirectional(LSTM(64, return_sequences=False, dropout=0.2)))

    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    model.summary()

    return model

# In[28]:


#Main

X, Y = load_data()

train_pearson_coeff = []
val_pearson_coeff = []
test_pearson_coeff = []
Y_train = []
Y_val = []
Y_test = []

VAL_loss = []


for fold in range(5):

    print('fold : '+str(fold+1))
    val_index = (fold+1)%5
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(X, Y, fold, val_index)
    
    num_train = len(x_train)
    num_val = len(x_val)
    num_test = len(x_test)
    
    #Normalizing Y
    y_train, y_val, y_test, mu_tr, sigma_tr = normalize_Y(y_train, y_val, y_test)

    fName = '/SavedModels/swbd/Ftr2_pch_interp/train_swbd_fold_' + str(fold+1)
    #building model
    model = build_model(num_filters, filter_length, pooling_size=2)
    
    print('Train on '+str(num_train)+' samples ')
    val_loss=[]
    flag=False

    for epoch in range(EPOCHS):
        
        if flag==True:
            break;

        print('Fold: '+str(fold+1))
        print('Epoch: '+str(epoch+1)) 
        count = 1
   
        index = np.random.permutation(num_train)

        for i in index:

            print('Training on fold-'+str(fold+1)+', epoch-'+str(epoch+1)+', sample-'+str(count))            
            count = count+1

            #reshaping to input in cnn
            x = np.resize(x_train[i],(1, x_train[i].shape[0], FEATURES))
            y = np.expand_dims(y_train[i], axis=0)
            history = model.fit(x, y)        
            
        if epoch>(EPOCHS_WITHOUT_VAL-1):
            val_pred = np.zeros(num_val)
            indx = np.random.permutation(num_val)
            for k in indx:
                #print('Testing on val : '+str(k+1))
                val_pred[k] = model.predict(np.resize(x_val[k],(1, x_val[k].shape[0], FEATURES)))
            cur_loss = (mean_squared_error(val_pred, y_val))
            val_loss = np.append(val_loss, cur_loss)
            print('val loss :' +str(val_loss))
            print('val_loss len :'+str(len(val_loss)))
            if len(val_loss)>2:
                if (val_loss[-3]<val_loss[-2]) and (val_loss[-2]<val_loss[-1]):
                    flag=True
                   
                  
    model.save(fName+'_.h5')
    model.save_weights(fName+'_weights.h5')

    #Predicting values for test data
    print('Test on '+str(num_test)+' samples ')
    test_pred = np.zeros(num_test)
    val_pred = np.zeros(num_test)
    for j in range(num_test):
        xv = np.resize(x_val[j],(1, x_val[j].shape[0], FEATURES))
        xt = np.resize(x_test[j],(1, x_test[j].shape[0], FEATURES))
        val_pred[j] = model.predict(xv)
        test_pred[j] = model.predict(xt)
    
    #Predictiong values for training data
    print('Predicting values for '+str(num_train)+' samples of training data ')
    train_pred = np.zeros(num_train)
    for j in range(num_train):
        x = np.resize(x_train[j],(1, x_train[j].shape[0], FEATURES))
        train_pred[j] = model.predict(x)
        
    #Denormalizing Y
    y_train = denormalize_Y(y_train, mu_tr, sigma_tr)
    train_pred = denormalize_Y(train_pred, mu_tr, sigma_tr)
    y_val = denormalize_Y(y_val, mu_tr, sigma_tr)
    val_pred = denormalize_Y(val_pred, mu_tr, sigma_tr)
    y_test = denormalize_Y(y_test, mu_tr, sigma_tr)
    test_pred = denormalize_Y(test_pred, mu_tr, sigma_tr)
    
    print('Calculating Pearson Coefficient.....')
    test_corr, _ = pearsonr(np.squeeze(test_pred), np.squeeze(y_test))
    test_pearson_coeff.append(test_corr)
    
    val_corr, _ = pearsonr(np.squeeze(val_pred), np.squeeze(y_val))
    val_pearson_coeff.append(val_corr)

    train_corr, _ = pearsonr(np.squeeze(train_pred), np.squeeze(y_train))
    train_pearson_coeff.append(train_corr)
    
    print('True Test Values : ' + str(y_test))
    print('Test Predictions : ' + str(test_pred))
    print('Test Pearson Coefficient : ' + str(test_pearson_coeff))

    print('True Val Values : ' + str(y_val))
    print('Val Predictions : ' + str(val_pred))
    print('Val Pearson Coefficient : ' + str(val_pearson_coeff))

    print('True Train Values : ' + str(y_train))
    print('Train Predictions : ' + str(train_pred))
    print('Train Pearson Coefficient : ' + str(train_pearson_coeff))

    
    #storing the results
    Y_test.append(test_pred)
    Y_val.append(val_pred)
    Y_train.append(train_pred)
    VAL_loss.append(val_loss)
    
test_pearson_coeff = np.array(test_pearson_coeff)
test_pearson_coeff_avg = np.sum(test_pearson_coeff)/5
print('Test Pearson Coefficient Avg: '+str(test_pearson_coeff_avg))

val_pearson_coeff = np.array(val_pearson_coeff)
val_pearson_coeff_avg = np.sum(val_pearson_coeff)/5
print('Val Pearson Coefficient Avg: '+str(val_pearson_coeff_avg))

train_pearson_coeff = np.array(train_pearson_coeff)
train_pearson_coeff_avg = np.sum(train_pearson_coeff)/5
print('Train Pearson Coefficient Avg: '+str(train_pearson_coeff_avg))


#fName_mat = '/Predictions/swbd_predictions/train_swbd'
#scipy.io.savemat(fName_mat+'_Ytest_pred', {'Ytest_pred': Y_test}, oned_as='row')
#scipy.io.savemat(fName_mat+'_test_coeff', mdict={'test_coeff': test_pearson_coeff}, oned_as='row')
#scipy.io.savemat(fName_mat+'_Yval_pred', {'Yval_pred': Y_val}, oned_as='row')
#scipy.io.savemat(fName_mat+'_val_coeff', mdict={'val_coeff': val_pearson_coeff}, oned_as='row')
#scipy.io.savemat(fName_mat+'_Ytrain_pred', {'Ytrain_pred': Y_train}, oned_as='row')
#scipy.io.savemat(fName_mat+'_train_coeff', mdict={'train_coeff': train_pearson_coeff}, oned_as='row')
#scipy.io.savemat(fName_mat+'_Val_loss', {'Val_loss': VAL_loss}, oned_as='row')



# In[ ]:





# In[ ]:




