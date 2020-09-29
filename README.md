# CNN-BLSTM Speaking Rate Estimator

This project contains coode for the Proposed Approach used in the paper "A robust speaking rate estimator using a CNN-BLSTM network"

## Description

This paper 

### Feature Computation

The input to the CNN-BLSTM model are 19 sub-band energies and pitch values. The tool used for computing these features is "Huckvale, M.: Speech filing system: Tools for speech research." and can be downloaded from http://www.phon.ucl.ac.uk/resource/sfs


#### 1. Files
This folder contains 3 files used for training the CNN-BLSTM model on the TIMIT and SWITCHBOARD dataset.
  
i.	train_train.py  - This file is used to train the CNN-BLSTM model on TIMIT dataset.
ii.	train_swbd.py  - This file is used to train the CNN-BLSTM model on SWITCHBOARD dataset.

iii.	train_timit_swbd.py - This file is used to train the CNN-BLSTM model on both TIMIT and SWITCHBOARD dataset.

iv. test.py - This file can be used to reproduce the  results using models present in "SavedModels" folder.

#### 2. SavedModels 
This folder contains the CNN-BLSTM models and their weights trained on the datasets.



## Usage

```python

pyhton3 test.py

```

## Contributors
