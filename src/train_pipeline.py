import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
import os
import pickle
from src import datasets

from src.config import config as c
import src.preprocessing.preprocessors as pp
from src.preprocessing.data_management import load_dataset, save_model, load_model

def initialize_data():
    data = load_dataset("train.csv")
    c.X_train = data.iloc[:, :-1].values 
    c.Y_train = data.iloc[:, -1].values.reshape(-1, 1)
    c.training_data = data

def training_data_generator():
    for i in range(c.training_data.shape[0] // c.mb_size):
        X_train_mb = c.X_train[i * c.mb_size:(i + 1) * c.mb_size, :]
        Y_train_mb = c.Y_train[i * c.mb_size:(i + 1) * c.mb_size]
        yield X_train_mb, Y_train_mb

datagen = training_data_generator()

def functional_mlp():
    inp = Input(shape=(c.X_train.shape[1],))
    first_hidden_out = Dense(units=4, activation="relu")(inp)
    second_hidden_out = Dense(units=2, activation="relu")(first_hidden_out)
    nn_out = Dense(units=1, activation="sigmoid")(second_hidden_out)
    return Model(inputs=inp, outputs=nn_out)

initialize_data()

functional_nn = functional_mlp()
functional_nn.compile(optimizer=c.optimizer, loss=tf.keras.losses.binary_crossentropy)

def binary_cross_entropy_loss(Y_hat, Y_true):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=Y_true, y_pred=Y_hat))

for e in range(c.epochs):
    for X_train_mb, Y_train_mb in training_data_generator():
        with tf.GradientTape() as tape:
            Y_pred = functional_nn(X_train_mb, training=True)
            loss_func = binary_cross_entropy_loss(Y_pred, Y_train_mb)
        gradients = tape.gradient(loss_func, functional_nn.trainable_weights)
        c.optimizer.apply_gradients(zip(gradients, functional_nn.trainable_weights))
    print("Epoch # {}, Loss Function Value = {}".format(e + 1, loss_func))

if __name__ == "__main__":
    save_model(functional_nn, c)