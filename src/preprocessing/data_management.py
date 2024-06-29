import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense

from src.config import config

def load_dataset(file_name):
    file_path = os.path.join(config.DATAPATH,file_name)
    data = pd.read_csv(file_path)
    return data

def save_model(model, config):
    pkl_file_path = os.path.join(config.SAVED_MODEL_PATH, "two_input_xor_nn.pkl")
    with open(pkl_file_path, "wb") as file_handle:
        pickle.dump(model.get_weights(), file_handle)

def load_model(file_name):
    pkl_file_path = os.path.join(config.SAVED_MODEL_PATH,file_name)

    with open(pkl_file_path,"rb") as file_handle:
        loaded_model = pickle.load(file_handle)
    
    return loaded_model