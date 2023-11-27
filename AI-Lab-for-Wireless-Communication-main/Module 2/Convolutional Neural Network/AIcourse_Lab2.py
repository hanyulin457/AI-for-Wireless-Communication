# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 08:59:07 2021

@author: Mint
"""

import random
import datetime
from functools import reduce
import scipy.io as sio
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
#%%
def read_csi_samples(filename,sample_num):
    data = sio.loadmat(filename)
    result = []
    for d in data['original_csi']:
        result.append(d[0]['csi'][0][0])
        result = result[0:sample_num]
    return np.abs(np.array(result).reshape((-1, Nt*Nr, Ns)).transpose((0, 2, 1)))
    #return np.abs(np.array(result).reshape((-1, Nt*Nr*Ns)))

def get_cnn_block(filter):
    block = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filter, 5, padding='same'),
            tf.keras.layers.MaxPool1D(padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.SpatialDropout1D(0.1),      
    ])
    return block
#%%

if __name__ == '__main__':
        
    #請更改路徑
    ROOT_DIR = 'G:\我的雲端硬碟\courses\大三\AI-Lab-for-Wireless-Communication\Convolutional Neural Network'
    
    Nap, Nt, Nr, Ns, M = 2, 2, 2, 56, 6
    dataset_num = 6*4
    
    """##Dataset"""
    
    train_database = [f'{ROOT_DIR}/data/{p}(mat)/{p}{i}_1' for i in range(1, 7) for p in ['db', 'a', 'c', 'j']]
    test_database = [f'{ROOT_DIR}/data/{p}(mat)/{p}{i}_2' for i in range(1, 7) for p in ['db', 'a', 'c', 'j']]
  
    
    sample_num = 600 #@param
    train_dataset = np.array([read_csi_samples(ds,sample_num) for ds in train_database])
    test_dataset = np.array([read_csi_samples(ds,sample_num) for ds in test_database])


    train_dataset = np.reshape(train_dataset,[dataset_num*sample_num,Ns,Nt*Nr])
    test_dataset = np.reshape(test_dataset,[dataset_num*sample_num,Ns,Nt*Nr])
    
    labels = np.zeros(dataset_num*sample_num)
    
    
    
    
    for i in range(dataset_num):
        for j in range(sample_num):
            temp = i % 6
            labels[(i-1)*sample_num+j] =  i % 6
    labels = np.reshape(labels,(dataset_num*sample_num,1))
#%%
    
    onehotencoder = OneHotEncoder()
    training_label = onehotencoder.fit_transform(labels).toarray()
    test_label = training_label
    
#%% 
    input_layer = tf.keras.layers.Input(shape=(Ns, Nt*Nr))
    
    c1 = tf.keras.layers.Conv1D(32, 5, padding='same', activation=tf.nn.leaky_relu)(input_layer)
    c2 = tf.keras.layers.Conv1D(64, 5, padding='same', activation=tf.nn.leaky_relu)(c1)
    c3 = tf.keras.layers.Conv1D(128, 5, padding='same', activation=tf.nn.leaky_relu)(c2)
    c4 = tf.keras.layers.Conv1D(128, 5, padding='same', activation=tf.nn.leaky_relu)(c3)
    
    f1 = tf.keras.layers.Flatten()(c4)
    
    d1 = tf.keras.layers.Dense(256, activation='relu')(f1)
    d2 = tf.keras.layers.Dense(64, activation='relu')(d1)

    output = tf.keras.layers.Dense(6, activation='softmax')(d2)
    
    model = tf.keras.models.Model(inputs=input_layer, outputs=output)
    model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=['accuracy'])
    model.summary()
    
    history = model.fit(train_dataset, training_label, validation_split=0.2, epochs=20, verbose=2)
    

    #%%
    #加分題 打亂資料順序
    
    #your code
    train_data = list(zip(train_dataset, training_label))

    # Shuffle the zipped data
    random.shuffle(train_data)
    
    # Unzip the shuffled data back into separate arrays
    train_dataset, training_label = zip(*train_data)
    
    # Convert back to numpy arrays
    train_dataset = np.array(train_dataset)
    training_label = np.array(training_label)
    
#%%


    #加分: Maxpooling/ batch normalize/ dropout
    #先完成基礎的CNN+DNN，再開始進行加分題

    inputs = tf.keras.layers.Input(shape=(Ns, Nt*Nr))
    #your code
    def get_cnn_block(filter):
        block = tf.keras.Sequential([
                tf.keras.layers.Conv1D(filter, 5, padding='same', activation=tf.nn.leaky_relu),
                tf.keras.layers.MaxPool1D(padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.SpatialDropout1D(0.1),      
        ])
        return block
    c1 = get_cnn_block(32)(inputs)
    c2 = get_cnn_block(64)(c1)
    c3 = get_cnn_block(128)(c2)
    c4 = get_cnn_block(128)(c3)
    
    f1 = tf.keras.layers.Flatten()(c4)
    
    d1 = tf.keras.layers.Dense(256, activation='relu')(f1)
    d2 = tf.keras.layers.Dense(64, activation='relu')(d1)

    output = tf.keras.layers.Dense(6, activation='softmax')(d2)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=['accuracy'])
    model.summary()
    
    history = model.fit(train_dataset, training_label, validation_split=0.2, epochs=20, verbose=2)
#%%    
    model.evaluate(test_dataset, test_label, batch_size = 256) 
    y_pred = onehotencoder.inverse_transform(model.predict(test_dataset))  
    y_true = onehotencoder.inverse_transform(test_label)    
    print('\nConfusion Matrix:')
    print(confusion_matrix(y_true,y_pred))

#%%
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()