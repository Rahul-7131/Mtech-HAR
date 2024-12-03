from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
#%matplotlib inline
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, concatenate, MaxPooling1D, Dropout, Reshape


# Load data
file_paths = [
    '/scratch/data/avinash1/pamap2/PAMAP2_Dataset/Protocol/subject101.dat',
    '/scratch/data/avinash1/pamap2/PAMAP2_Dataset/Protocol/subject102.dat',
    '/scratch/data/avinash1/pamap2/PAMAP2_Dataset/Protocol/subject103.dat',
    '/scratch/data/avinash1/pamap2/PAMAP2_Dataset/Protocol/subject104.dat',
    '/scratch/data/avinash1/pamap2/PAMAP2_Dataset/Protocol/subject105.dat',
    '/scratch/data/avinash1/pamap2/PAMAP2_Dataset/Protocol/subject106.dat',
    '/scratch/data/avinash1/pamap2/PAMAP2_Dataset/Protocol/subject107.dat',
    '/scratch/data/avinash1/pamap2/PAMAP2_Dataset/Protocol/subject108.dat',
    '/scratch/data/avinash1/pamap2/PAMAP2_Dataset/Protocol/subject109.dat'
    # Add paths to other subject files as needed
]

subjectID = [1,2,3,4,5,6,7,8,9]

activityIDdict = {0: 'transient',
              1: 'lying',
              2: 'sitting',
              3: 'standing',
              4: 'walking',
              5: 'running',
              6: 'cycling',
              7: 'Nordic_walking',
              9: 'watching_TV',
              10: 'computer_work',
              11: 'car driving',
              12: 'ascending_stairs',
              13: 'descending_stairs',
              16: 'vacuum_cleaning',
              17: 'ironing',
              18: 'folding_laundry',
              19: 'house_cleaning',
              20: 'playing_soccer',
              24: 'rope_jumping' }

colNames = ["timestamp", "activityID","heartrate"]

IMUhand = ['handTemperature', 
           'handAcc16_1', 'handAcc16_2', 'handAcc16_3', 
           'handAcc6_1', 'handAcc6_2', 'handAcc6_3', 
           'handGyro1', 'handGyro2', 'handGyro3', 
           'handMagne1', 'handMagne2', 'handMagne3',
           'handOrientation1', 'handOrientation2', 'handOrientation3', 'handOrientation4']

IMUchest = ['chestTemperature', 
           'chestAcc16_1', 'chestAcc16_2', 'chestAcc16_3', 
           'chestAcc6_1', 'chestAcc6_2', 'chestAcc6_3', 
           'chestGyro1', 'chestGyro2', 'chestGyro3', 
           'chestMagne1', 'chestMagne2', 'chestMagne3',
           'chestOrientation1', 'chestOrientation2', 'chestOrientation3', 'chestOrientation4']

IMUankle = ['ankleTemperature', 
           'ankleAcc16_1', 'ankleAcc16_2', 'ankleAcc16_3', 
           'ankleAcc6_1', 'ankleAcc6_2', 'ankleAcc6_3', 
           'ankleGyro1', 'ankleGyro2', 'ankleGyro3', 
           'ankleMagne1', 'ankleMagne2', 'ankleMagne3',
           'ankleOrientation1', 'ankleOrientation2', 'ankleOrientation3', 'ankleOrientation4']

columns = colNames + IMUhand + IMUchest + IMUankle  #all columns in one list

#len(columns)


dataCollection = pd.DataFrame()

for file in file_paths:
    procData = pd.read_table(file, header=None, sep='\s+')
    procData.columns = columns
    procData['subject_id'] = int(file[-5])
    dataCollection = pd.concat([dataCollection, procData], ignore_index=True)

dataCollection.reset_index(drop=True, inplace=True)
#dataCollection.head()

def dataCleaning(dataCollection):
        dataCollection = dataCollection.drop(['handOrientation1', 'handOrientation2', 'handOrientation3', 'handOrientation4',
                                             'chestOrientation1', 'chestOrientation2', 'chestOrientation3', 'chestOrientation4',
                                             'ankleOrientation1', 'ankleOrientation2', 'ankleOrientation3', 'ankleOrientation4'],
                                             axis = 1)  # removal of orientation columns as they are not needed
        dataCollection = dataCollection.drop(dataCollection[dataCollection.activityID == 0].index) #removal of any row of activity 0 as it is transient activity which it is not used
        dataCollection = dataCollection.apply(pd.to_numeric, errors = 'coerce') #removal of non numeric data in cells
        dataCollection = dataCollection.interpolate() #removal of any remaining NaN value cells by constructing new data points in known set of data points
        
        return dataCollection


dataCol = dataCleaning(dataCollection)
dataCol.reset_index(drop = True, inplace = True)
#dataCol.head(10)

dataCol.isnull().sum()
for i in range(0,4):
    dataCol["heartrate"].iloc[i]=100


train_df = dataCol.sample(frac=0.8, random_state=1)
test_df = dataCol.drop(train_df.index)

#train_df.describe()

train_df = train_df.drop(["timestamp", "subject_id"],axis=1)

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler,RobustScaler

#scaling all columns except subject and activity 
scaler = RobustScaler()
df_scaled = train_df.copy()
df_scaled_test = test_df.copy()

df_scaled.iloc[:,1:41] = scaler.fit_transform(df_scaled.iloc[:,1:41])
df_scaled_test.iloc[:,1:41] = scaler.fit_transform(df_scaled_test.iloc[:,1:41])

#df_scaled.head()

X_train = df_scaled.drop('activityID', axis=1).values
y_train = df_scaled['activityID'].values

# Test Dataset
X_test = df_scaled.drop('activityID', axis=1).values
y_test = df_scaled['activityID'].values

from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X_train)
var= pca.explained_variance_ratio_
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
pca = PCA(n_components=17)
X_train=pca.fit_transform(X_train)
X_test=pca.fit_transform(X_test)

# Reshape data
X_train_reshaped = np.expand_dims(X_train, axis=-1)
X_test_reshaped = np.expand_dims(X_test, axis=-1)

unique_classes = [0, 1, 2, 3, 17, 16, 12, 13, 4, 7, 6, 5, 24]
class_mapping = {old: new for new, old in enumerate(unique_classes)} #sequentially mapped for one_hot_encoding
y_train_mapped = [class_mapping[label] for label in y_train]
y_test_mapped = [class_mapping[label] for label in y_test]

# One-hot encoding 
num_classes = 13
y_train_encoded = tf.keras.utils.to_categorical(y_train_mapped, num_classes)
y_test_encoded = tf.keras.utils.to_categorical(y_test_mapped, num_classes)


use_subset = True

if use_subset:
    X_train_subset, _, y_train_subset, _ = train_test_split(X_train_reshaped, y_train_encoded, test_size=0.8, random_state=42)
    X_test_subset, _, y_test_subset, _ = train_test_split(X_test_reshaped, y_test_encoded, test_size=0.8, random_state=42)
    using_train_x = X_train_subset
    using_train_y = y_train_subset
    using_test_x = X_test_subset
    using_test_y = y_test_subset
else:
    using_train_x = X_train_reshaped
    using_train_y = y_train_encoded
    using_test_x = X_test_reshaped
    using_test_y = y_test_encoded


'''C-LSTM'''
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, concatenate, MaxPooling1D, Dropout, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def create_c_lstm(input_shape,num_classes):
  inputs = Input(shape=input_shape)
  conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
  conv2 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
  conv3 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
  concat_conv = concatenate([conv1, conv2, conv3], axis=-1)
  reshape = Reshape((concat_conv.shape[1], -1))(concat_conv)
  lstm = LSTM(128)(reshape)
  dense = Dense(64, activation='tanh')(lstm)
  outputs = Dense(num_classes, activation='softmax')(dense)

  model = Model(inputs, outputs)
  return model


# Model creation
c_lstm_model = create_c_lstm(using_train_x.shape[1:], num_classes)
c_lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['mae'])
c_lstm_model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Training
history = c_lstm_model.fit(using_train_x, using_train_y, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping,reduce_lr])

# Evaluation
y_pred = c_lstm_model.predict(using_test_x)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(using_test_y, axis=1)

accuracy = accuracy_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes, average='weighted')
recall = recall_score(y_true, y_pred_classes, average='weighted')
f1 = f1_score(y_true, y_pred_classes, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Inference time
start_time = time.time()
predictions = c_lstm_model.predict(using_test_x)
end_time = time.time()
inference_time = end_time - start_time
print(f'Inference Time(CMC-LSTM): {inference_time} seconds')


