import os
import pathlib
import src
from keras.optimizers import RMSprop

LOSS_FUNCTION = "binary_cross_entropy_loss"
mb_size = 2
epochs = 100
optimizer = RMSprop(learning_rate=0.01)

X_train = None
Y_train = None
training_data = None

PACKAGE_ROOT = pathlib.Path(src.__file__).resolve().parent
DATAPATH = os.path.join(PACKAGE_ROOT,"datasets")

SAVED_MODEL_PATH = os.path.join(PACKAGE_ROOT,"trained_models")