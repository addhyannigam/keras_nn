import pandas as pd
import numpy as np
import tensorflow as tf
import train_pipeline as tp

X_test = pd.DataFrame(data={"x1": [0, 0, 1, 1], "x2": [0, 1, 0, 1]})


predictions = tp.functional_nn.predict(X_test)
print("Predictions:")
print(predictions)

# Rounding predictions to get binary output
binary_predictions = (predictions > 0.5).astype(int)
print("Binary Predictions:")
print(binary_predictions)