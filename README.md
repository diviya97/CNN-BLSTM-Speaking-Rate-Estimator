# CNN-BLSTM-Speaking-Rate-Estimator

1. DataFileNames - This folder contains the names of the individual speech spursts fot TIMIT and SWITCHBOARD dataset. The data was loaded in the order given in these files for splitting into 5 folds.

2. Files - This folder contains 3 files used for training the CNN-BLSTM model on the TIMIT and SWITCHBOARD dataset.
  
  i) timit_Ftr2_pch_interp_SampleBySample_with_val.py  - This file is used to train the CNN-BLSTM model on TIMIT dataset.
  ii) swbd_timit_model_Ftr2_pch_interp_SampleBySample_with_val.py  - This file is used to train the CNN-BLSTM model on SWITCHBOARD dataset.
  iii) Ftr2_pch_interp_train_using_timit_and_swbd_data.py - This file is used to train the CNN-BLSTM model on both TIMIT and SWITCHBOARD dataset.
 
3. SavedModels - This folder comtains the CNN_BLSTM models  and their weights trained on the datasets. 
  

  
