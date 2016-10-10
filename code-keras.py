# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 20:55:07 2016

@author: QiuXun

Building single model with neural networks for TalkingData competition@Kaggle

Features used:
- phone brand/device_model/foreign phone brand/new phone brand/popular brand
- bag of apps installed
- bag of category for installed app
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.advanced_activations import PReLU
from keras.layers import Dense, Dropout

# set random seed
seed = 314
np.random.seed(seed)

#----------------- Auxiliary functions ----------------------------------------

# batch data generator that convert only the current batch to dense array
def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator 
    # (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0


#-------------------Data processing -------------------------------------------
####################
#   1. Phone Brand
####################
print("# Read device_genderage file")
# Device info is pre-processed in code-xgb.R
pbd_new = pd.read_csv('device_genderage_utf8.csv', dtype={'device_id': np.str})

##################
#   2. Apps
##################
print("## Processing info on apps")
app_events = pd.read_csv('app_events.csv', dtype={'device_id' : np.str})

# concatenate unique app_id for each event_id
app_events= app_events.groupby("event_id")["app_id"].apply(
    lambda x: " ".join(set("app_id:" + str(s) for s in x)))

events = pd.read_csv('events.csv', dtype={'device_id': np.str})
events["app_id"] = events["event_id"].map(app_events)
events = events.dropna()
del app_events

events = events[["device_id", "app_id"]]

# concatenate unique app_id for each device_id
events = events.groupby("device_id")["app_id"].apply(
    lambda x: " ".join(set(str(" ".join(str(s) for s in x)).split(" "))))
events = events.reset_index(name="app_id")

# expand to multiple rows, each row is one app_id and one device_id
events = pd.concat([pd.Series(row['device_id'], row['app_id'].split(' '))
                    for _, row in events.iterrows()]).reset_index()
events.columns = ['app_id', 'device_id']
device_app = events[["device_id", "app_id"]]

device_app.to_csv('device_installedapp.csv', header=True)


####################
#   3. App categories
####################
print("## Processing info on app categories")
app_labels = pd.read_csv('app_labels.csv')
label_cat = pd.read_csv('label_categories.csv')

# each row is one app_id with one of its label_id and the category
app_labels = app_labels.merge(label_cat,on='label_id',how='left')

# each row is one app_id with one of its category and its occurence
app_labels = app_labels.groupby(["app_id","category"]).agg('size').reset_index()
# each row is one app_id with one of its category
app_labels = app_labels[['app_id','category']]

events['app_id'] = events['app_id'].map(lambda x : x.lstrip('app_id:'))
events['app_id'] = events['app_id'].astype(str)
app_labels['app_id'] = app_labels['app_id'].astype(str)

# each row is one device_id with one of its app_id and one of its category
events= pd.merge(events, app_labels, on = 'app_id',how='left').astype(str)

# removing app_id and keep the occurence of category for each device_id
events= events.groupby(["device_id","category"]).agg('size').reset_index()
events= events[['device_id','category']]

device_category = events[["device_id", "category"]]    

device_category.to_csv('device_category.csv', header=True)


####################
#  4. Train and Test
####################
print("# Processing Train and Test set")

train = pd.read_csv('gender_age_train.csv',
                    dtype={'device_id': np.str})
train.drop(["age", "gender"], axis=1, inplace=True)

test = pd.read_csv('gender_age_test.csv',
                    dtype={'device_id': np.str})
test["group"] = np.nan

# Group Labels
Y = train["group"]
label_group = LabelEncoder()
Y = label_group.fit_transform(Y)
device_id_test = test["device_id"]

Df = pd.concat((train, test), axis=0, ignore_index=True)

Df = pd.merge(Df, pbd_new, how="left", on="device_id")
Df["phone_brand"] = Df["phone_brand"].apply(lambda x: "phone_brand:" + str(x))
Df["device_model_new"] = Df["device_model_new"].apply(
    lambda x : "device_model_new:" + str(x))
Df["phonebrand_group1"] = Df["phonebrand_group1"].apply(
    lambda x : "phonebrand_group1:" + str(x))
Df["phonebrand_foreign"] = Df["phonebrand_foreign"].apply(
    lambda x : "phonebrand_foreign:" + str(x))
Df["phonebrand_old"] = Df["phonebrand_old"].apply(
    lambda x : "phonebrand_old:" + str(x))

#######################
#  5. Concat Features
#######################

print("## Concat all features")

f1 = Df[["device_id", "phone_brand"]]   
f2 = Df[["device_id", "device_model_new"]]  
f3 = Df[["device_id", "phonebrand_group1"]]  
f4 = Df[["device_id", "phonebrand_foreign"]] 
f5 = Df[["device_id", "phonebrand_old"]]  

Df = None

## Xun: read in saved data as output from previous steps
f6 = pd.read_csv('device_category.csv', dtype={'device_id' : np.str})
f6 = f6[["device_id","category"]]
f7 = pd.read_csv('device_installedapp.csv', dtype={'device_id' : np.str})
f7 = f7[["device_id","app_id"]]

f1.columns.values[1] = "feature"
f2.columns.values[1] = "feature"
f3.columns.values[1] = "feature"
f4.columns.values[1] = "feature"
f5.columns.values[1] = "feature"
f6.columns.values[1] = "feature"
f7.columns.values[1] = "feature"

FLS = pd.concat((f1,f2,f3,f4,f5,f6,f7), axis=0, ignore_index=True)

#############################
# 6. Construct sparse matrix
#############################
print("## Constructing sparse matrix")

device_ids = FLS["device_id"].unique()
feature_cs = FLS["feature"].unique()

data = np.ones(len(FLS))

dec = LabelEncoder().fit(FLS["device_id"])
row = dec.transform(FLS["device_id"])
col = LabelEncoder().fit_transform(FLS["feature"])
sparse_matrix = sparse.csr_matrix(
    (data, (row, col)), shape=(len(device_ids), len(feature_cs)))

# level reduction
train_row = dec.transform(train["device_id"])
sparse_matrix_train = sparse_matrix[train_row,:]
sparse_matrix = sparse_matrix[:, sparse_matrix_train.getnnz(0) > 0]

del FLS, data, f1, f2, f3, f4, f5, f6, f7, events

#############################
#  7. Train/test data split
#############################
print("# Split train/test data")

# sparse matrix for training data
train_row = dec.transform(train["device_id"])
train_sp = sparse_matrix[train_row, :]

# sparse matrix for test data
test_row = dec.transform(test["device_id"])
test_sp = sparse_matrix[test_row, :]

X_train, X_val, y_train, y_val = train_test_split(
    train_sp, Y, train_size=0.7, random_state=10)

#------------------- Model training and prediciton-----------------------------
######################
#  8. Model training
######################

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dropout(0.2,input_shape=(X_train.shape[1],)))   
    model.add(Dense(150, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(50, input_dim=X_train.shape[1], init='normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(12, init='normal', activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer='adadelta', 
                  metrics=['accuracy']) 
    return model

model=baseline_model()

print("## Model training")

fit= model.fit_generator(generator=batch_generator(X_train,y_train,400,True),
                         nb_epoch=27,
                         samples_per_epoch=X_train.shape[0],
                         validation_data=(X_val.todense(), y_val), verbose=2
                         )


######################
#  9. Prediction
######################
print("## Final prediction")
probs = model.predict_generator(generator=batch_generatorp(test_sp, 800), 
                                 val_samples=test_sp.shape[0])
prediction = pd.DataFrame(probs , columns=label_group.classes_)
prediction["device_id"] = device_id_test
prediction = prediction.set_index("device_id")

prediction.to_csv('submission_keras.csv', index=True, index_label='device_id')

print("Done")