import os
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from time import time
from PIL import Image
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from DeepLearningModels.my_classes import  DataGenerator
from keras.callbacks import TensorBoard


def image_array(df):
    for i in range(len(df)):
        new_path = path + 'map_images/' + df.loc[i, 'image_id']
        temp = Image.open(new_path)
        img_data = np.array(temp)
        name = path + 'map_images/' + (df.loc[i, 'image_id'].split('.')[0] + '.npy')
        np.save(name, img_data)
        del [img_data, name, new_path, temp]


def path_change(df):
    for i in range(len(df)):
        new_path = path +'map_images/'+ df.loc[i, 'image_id']
        name = path+'map_images/'+(df.loc[i, 'image_id'].split('.')[0]+'.npy')
        df.loc[i, 'image_id'] = name
        del [name, new_path]


# Put your root path here
'''*******************************************************
DeepLearningModels/
                |____ keras_script.py
                |____ my_classes.py
                |____ my_images/ : all images here
********************************************************'''
path = 'D:/myProject/DeepLearningModels/'
# Reading csv into Dataframe
df_data = pd.read_csv(path+'map_traininglabels.csv')
#image_array(df_data)    # converting image files to ndarray
path_change(df_data)    # saving new path of data files
X_train, X_test, y_train, y_test = train_test_split(df_data['image_id'], df_data['Class'], test_size=0.25)

# Parameters
params = {'dim': (256, 256), 'batch_size': 8, 'n_classes': 2, 'n_channels': 3, 'shuffle': True}

# Datasets
partition = {'train': X_train.tolist(), 'validation': X_test.tolist()}
df_ = df_data
df_ = df_.drop(['score'], 1)
labels = df_.set_index('image_id').T.to_dict('records')[0]
del [df_data , df_, X_train, X_test, y_train, y_test]

# Generators

training_generator = DataGenerator(partition['train'], labels, batch_size=8, dim=(256,256), n_channels=3, n_classes=2, shuffle=True)
validation_generator = DataGenerator(partition['validation'], labels, batch_size=8, dim=(256, 256), n_channels=3, n_classes=2, shuffle=True)

# Design model
model = Sequential()
# Architecture
model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(2, activation='sigmoid'))

model.summary()

# metrics involved in data
#print(model.metrics_names)

# Model Compilation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# tensorboard callback
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
# Train model on dataset
"""***************************************************************************
    def fit_generator(self, generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0):
    ----------Trains the model on data generated batch-by-batch by a Python generator(or an instance of `Sequence`)-------------.
        The generator is run in parallel to the model, for efficiency. Suited if you have low memory

        The use of `keras.utils.Sequence` guarantees the ordering and guarantees the single use of every input per epoch
         when using `use_multiprocessing=True`.

        # Arguments
        *generator: A generator or an instance of `Sequence`(`keras.utils.Sequence`) object in order to avoid
                duplicate data when using multiprocessing.
                The output of the generator must be either - a tuple `(inputs, targets)` - a tuple `(inputs, targets, sample_weights)`.
        *steps_per_epoch: Integer. Total number of steps (batches of samples) to yield from `generator` before declaring one epoch
                Optional for `Sequence`: if unspecified, will use the `len(generator)` as a number of steps.
                by default steps_per_epoch= len(tarining)/batch_size
        *epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire data provided, as defined by `steps_per_epoch`.
        *verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        *callbacks: List of `keras.callbacks.Callback` instances.
        *validation_data: This can be either - a generator or a `Sequence` object for the validation data
                - tuple `(x_val, y_val)`  - tuple `(x_val, y_val, val_sample_weights)` on which to evaluate the loss and any model metrics at the end of each epoch.
        *validation_steps: Only relevant if `validation_data` is a generator. Total number of steps (batches of samples) to yield from `validation_data` generator before stopping
                at the end of every epoch. It should typically be equal to the number of samples of your validation dataset divided by the batch size.
        *class_weight: Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). 
        *max_queue_size: Integer. Maximum size for the generator queue. If unspecified, `max_queue_size` will default to 10.
        *workers: Integer. Maximum number of processes to spin up when using process-based threading.
                Default = 1. If 0, will execute the generator on the main thread.
        *use_multiprocessing: Boolean. If `True`, use process-based threading. Default = 'False'
                Note that because this implementation relies on multiprocessing,you should not pass non-picklable arguments to the generator
                as they can't be passed easily to children processes.
        *shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch. 
        *initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).

        # Returns
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).

        # Raises
            ValueError: In case the generator yields data in an invalid format.

        # Example

        ```python
        def generate_arrays_from_file(path):
            while True:
                with open(path) as f:
                    for line in f:
                        # create numpy arrays of input data
                        # and labels, from each line in the file
                        x1, x2, y = process_line(line)
                        yield ({'input_1': x1, 'input_2': x2}, {'output': y})

        model.fit_generator(generate_arrays_from_file('/my_file.txt'),
                            steps_per_epoch=10000, epochs=10)
        ```
        
        return training_generator.fit_generator(self, generator, steps_per_epoch=steps_per_epoch,
            epochs=epochs, verbose=verbose, callbacks=callbacks, validation_data=validation_data,
            validation_steps=validation_steps, class_weight=class_weight, max_queue_size=max_queue_size,
            workers=workers, use_multiprocessing=use_multiprocessing, shuffle=shuffle, initial_epoch=initial_epoch)
***************************************************************************"""
model.fit_generator(generator=training_generator, epochs=2, validation_data=validation_generator, callbacks=[tensorboard])
print("Training Completed")
scores = model.evaluate_generator(generator=validation_generator, verbose=1)
print("Evaluation completed")
scores_ = model.predict_generator(generator=validation_generator, verbose=1)
print("predicted on test data")