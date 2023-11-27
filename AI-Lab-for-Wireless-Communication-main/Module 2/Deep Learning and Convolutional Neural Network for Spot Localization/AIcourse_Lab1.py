# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#下載mnist資料集檔案 資料集檔案位置:C:\Users\.keras\datasets\mnist.npz
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
np.random.seed(10)

#%%
if __name__ == '__main__':
    #load data
    (x_train_image,y_train_label),\
    (x_test_image,y_test_label) = mnist.load_data()
    
    #reshape
    x_train_imre = x_train_image.reshape(x_train_image.shape[0], x_train_image.shape[1]*x_train_image.shape[2]).astype('float32')
    x_test_imre = x_test_image.reshape(x_test_image.shape[0], x_test_image.shape[1]*x_test_image.shape[2]).astype('float32')
    
    #normalize
    x_train_normalize = x_train_imre/255
    x_test_normalize = x_test_imre/255
    
    #one hot
    y_TrainOneHot = np_utils.to_categorical(y_train_label)
    y_TestOneHot = np_utils.to_categorical(y_test_label)
    
    #%% 加分 畫train data的圖
    # plt.imshow
    # your code
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10,4))
    axes = axes.ravel()
    
    for i in np.arange(0, 10):
        axes[i].imshow(x_train_image[y_train_label == i][0], cmap='gray')
        axes[i].set_title("Number: " + str(i))
        axes[i].axis('off')
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    
    #%%
    #build model
    inputs = keras.Input(shape = 784) 
    # your code
    d1 = tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu)(inputs)
    d2 = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu)(d1)
    d3 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)(d2)
    d4 = tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu)(d3)
    outputs = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(d4)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss="MSE", optimizer="Adam", metrics=['accuracy'])
    
    #start training
    train_history = model.fit(x=x_train_normalize, y=y_TrainOneHot, validation_split=0.2, epochs=10, batch_size=200, verbose=2)
    
    def show_train_history(train_history, train, validation):
        plt.plot(train_history.history[train])
        plt.plot(train_history.history[validation])
        plt.title('train history')
        plt.ylabel(train)
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
    show_train_history(train_history, 'accuracy', 'val_accuracy')
    show_train_history(train_history, 'loss', 'val_loss')
    
    scores = model.evaluate(x_test_normalize, y_TestOneHot)
    print('test loss, test accuracy=', scores)
    
    
    #%% confusion matrix
    
    # your code
    y_pred_onehot = model.predict(x_test_normalize)
    y_pred = np.argmax(y_pred_onehot, axis=1)
    y_test = np.argmax(y_TestOneHot, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks(np.arange(10))