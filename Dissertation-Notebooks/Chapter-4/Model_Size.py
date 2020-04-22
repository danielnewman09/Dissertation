import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import psutil
import time
from joblib import dump, load

gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

process = psutil.Process(os.getpid())
basePath = '/home/dnewman/'
print(process.memory_info().rss)  # in bytes 

current_memory = process.memory_info().rss
time.sleep(5)
model = load(basePath + "Models/GMM/PCA-GMM.joblib")
time.sleep(5)
loaded_memory = process.memory_info().rss

model_memory = loaded_memory - current_memory
print(model_memory)

current_memory = process.memory_info().rss
time.sleep(5)
model = load_model(basePath + "Models/Autoencoder/Full/CNN-AE.h5")
time.sleep(5)
loaded_memory = process.memory_info().rss

model_memory = loaded_memory - current_memory
print(model_memory)

current_memory = process.memory_info().rss
time.sleep(5)
model = load_model(basePath + "Models/Autoencoder/Full/AE.h5")
time.sleep(5)
loaded_memory = process.memory_info().rss

model_memory = loaded_memory - current_memory
print(model_memory)

current_memory = process.memory_info().rss
time.sleep(5)
model = tf.lite.Interpreter(model_path=basePath + "Models/Autoencoder/Lite/CNN-AE-Lite.tflite")
time.sleep(5)
model.allocate_tensors()
time.sleep(5)
loaded_memory = process.memory_info().rss

model_memory = loaded_memory - current_memory
print(model_memory)

current_memory = process.memory_info().rss
time.sleep(5)
model = load(basePath + "Models/GNB/PCA-GNB.joblib")
time.sleep(5)
loaded_memory = process.memory_info().rss

model_memory = loaded_memory - current_memory
print(model_memory)

current_memory = process.memory_info().rss
time.sleep(5)
model = load_model(basePath + "Models/MLP-Classifier/Full/MLP.h5")
time.sleep(5)
loaded_memory = process.memory_info().rss

model_memory = loaded_memory - current_memory
print(model_memory)

current_memory = process.memory_info().rss
time.sleep(5)
model = load_model(basePath + "Models/MLP-Classifier/Full/CNN-MLP.h5")
time.sleep(5)
loaded_memory = process.memory_info().rss

model_memory = loaded_memory - current_memory
print(model_memory)

current_memory = process.memory_info().rss
time.sleep(5)
model = tf.lite.Interpreter(model_path=basePath + "Models/MLP-Classifier/Lite/CNN-MLP-Lite.tflite")
time.sleep(5)
model.allocate_tensors()
time.sleep(5)
loaded_memory = process.memory_info().rss

model_memory = loaded_memory - current_memory
print(model_memory)
