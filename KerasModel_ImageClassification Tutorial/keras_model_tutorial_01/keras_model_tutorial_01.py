'''
Objective: To seperate the given images into training and testing dataset
build a classification network
find the confusion matrix

The image dataset can be downloaded from https://drive.google.com/file/d/18-XchvcQcmLMkNXNoiFruZfx340hnaUR/view?usp=sharing
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Path for images
path = '/home/ankit/DeepModels_Python/'

#reading the csv
df_label = pd.read_csv(path+'map_traininglabels.csv')
categories = (df_label['Class'].unique())  #gives us total number of classes=2(here)
print(categories)
count = df_label['Class'].value_counts()   #gives number of images in each class
print(count)
del [count]
img_lst = []


def seperating_class_dir(df):
    """*******************************************************************
    this function segregates the images in folders of respective class

    *******************************************************************"""
    for i in range(len(df)):
        if df.loc[i,'Class']==1:
            if not os.path.exists(path+'1'):
                os.mkdir(path+'1')
            temp = cv2.imread(path+'map_images/'+df.loc[i,'image_id'])

            cv2.imwrite((path+'1/'+df.loc[i,'image_id']), temp)
        else:
            if not os.path.exists(path+'0'):
                os.mkdir(path+'0')
            temp = cv2.imread(path + 'images/' + df.loc[i, 'image_id'])
            cv2.imwrite((path + '0/' + df.loc[i, 'image_id']), temp)


def img_path(df):
    '''***********************************************
    add complete image path to the 'image_id' colcumn in the datframe
    ***********************************************'''
    for i in range(len(df)):
        new_path = path+'images/'+df.loc[i,'image_id']
        df.loc[i,'image_id']=new_path


def img_val(df):
    '''***********************************************
    reads image using openCV and appends the data to list
    ***********************************************'''
    for i in (range(len(df))):
        temp=cv2.imread(df.loc[i, 'image_id'])
        img_lst.append(temp)
        del temp
        
#seperating_class_dir(df_label)
img_path(df_label)
img_val(df_label)

# converting list to ndarray
img_lst = np.array(img_lst)
y = np.array(df_label['Class'])

X_train, X_test, y_train, y_test = train_test_split(img_lst, y, test_size=0.25)

    
'''**********************************************
Keras Neyral Architecture for Training
********************************************I'''
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

'''******************************* model fitting using "model.fit" **************************************************
-------------Trains the model for a given number of epochs (iterations on a dataset)----------
model.fit(self,
            x=None, y=None, batch_size=None,
            epochs=1, verbose=1, callbacks=None,
            validation_split=0., validation_data=None,
            shuffle=True, class_weight=None,
            sample_weight=None, initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None, **kwargs)

---------
 # Arguments
        *x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        *y: Numpy array of target/label data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        *batch_size: Integer or `None`. Number of samples per gradient update. Default =32
        *epochs: Integer. Number of epochs to train the model.
        *verbose: Integer. 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
        *callbacks: List of `keras.callbacks.Callback` instances.
        *validation_split: Float between 0 and 1., Fraction of the training data to be used as validation data.
        *validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate
                the loss and any model metrics at the end of each epoch.
                The model will not be trained on this data.`validation_data` will override `validation_split`.
        *shuffle: Boolean (whether to shuffle the training data before each epoch) or str (for 'batch').
                'batch' is a special option for dealing with the limitations of HDF5 data.
        *class_weight: Optional dictionary mapping class indices (integers) to a weight (float) value,
        *sample_weight: Optional Numpy array of weights forthe training samples
        *initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
        *steps_per_epoch: Integer or `None`. Total number of steps (batches of samples) before declaring one epoch finished and starting the
                next epoch.
        *validation_steps: Only relevant if `steps_per_epoch` is specified. Total number of steps (batches of samples)
                to validate before stopping.
'''
model.fit(X_train, y_train, epochs=2, batch_size=2)


# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)

# serialize weights to HDF5
model.save_weights(path + "/model.h5")
print("Saved model to disk")
predictions = model.predict(X_test)
predictions = (predictions > 0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)




























