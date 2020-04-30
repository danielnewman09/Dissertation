#!flask/bin/python

import os
import numpy as np

from scipy import signal
from scipy.stats import describe

from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import mean_squared_error
#import tflite_runtime.interpreter as tflite
import tensorflow as tf
# import tensorflow.keras as keras
# from Custom_Layers import Dropout_Live

from joblib import dump, load


cwd = os.path.dirname(os.path.abspath(__file__))    

class RunModels(object):
    def __init__(self,preload_models,basePath,desktop=False):

        self.basePath = basePath
        self.preload_models = preload_models

        if preload_models == True:

            self.pca_gmm_model = load(basePath + "Models/GMM/PCA-GMM.joblib")
            #self.cnn_ae_model = load_model(basePath + "Models/Autoencoder/Full/CNN-AE.h5")
            #self.ae_model = load_model(basePath + "Models/Autoencoder/Full/AE.h5")
            self.cnn_ae_lite_model = tf.lite.Interpreter(model_path=self.basePath + "Models/Autoencoder/Lite/CNN-AE-Lite.tflite")

            self.pca_gnb_model = load(basePath + "Models/GNB/PCA-GNB.joblib")
            #self.mlp_model = load_model(basePath + "Models/MLP-Classifier/Full/MLP.h5")
            #self.cnn_mlp_model = load_model(basePath + "Models/MLP-Classifier/Full/CNN-MLP.h5")
            self.cnn_mlp_lite_model = tf.lite.Interpreter(model_path=self.basePath + "Models/MLP-Classifier/Lite/CNN-MLP-Lite.tflite")
       


    def classifier_inference_full(self,values,modelId):

        xInference = np.atleast_2d(np.array(values).astype(np.float32))

        if self.preload_models:

            if 'cnn' in modelId.lower():
                model = self.cnn_mlp_model
            else:
                model = self.mlp_model
        else:
            if 'cnn' in modelId.lower():
                model = load_model(self.basePath + "Models/MLP-Classifier/Full/CNN-MLP.h5")
            else:
                model = load_model(self.basePath + "Models/MLP-Classifier/Full/MLP.h5")

        X_predict = np.atleast_2d(xInference)

        if 'cnn' in modelId.lower():
            X_predict = X_predict[...,np.newaxis]

        predict = model.predict(X_predict)
        classification = predict[0,0].astype(float)

        return classification

    def model_inference_full(self,values,modelId):

        xInference = np.atleast_2d(np.array(values).astype(np.float32))

        if self.preload_models == True:

            if 'cnn' in modelId.lower():
                model = self.cnn_ae_model
            else:
                model = self.ae_model
        else:
            if 'cnn' in modelId.lower():
                model = load_model(self.basePath + "Models/Autoencoder/Full/CNN-AE.h5")
            else:
                model = load_model(self.basePath + "Models/Autoencoder/Full/AE.h5")

        X_predict = np.atleast_2d(xInference)

        if 'cnn' in modelId.lower():
            X_predict = X_predict[...,np.newaxis]

        predict = model.predict(X_predict)
        mse = mean_squared_error(X_predict,predict).numpy().flatten()[0].astype(float)

        return mse

    def model_inference_lite(self,values,modelId):

        xInference = np.atleast_2d(np.array(values).astype(np.float32))

        if self.preload_models == True:
            interpreter = self.cnn_ae_lite_model
        else:
            interpreter = tflite.Interpreter(model_path=self.basePath + "Models/Autoencoder/Lite/CNN-AE-Lite.tflite")

        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test model on random input data.
        input_shape = input_details[0]['shape']
        # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        input_data = xInference.reshape(input_shape).astype(np.float32)

        num_samples = 1

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index']).reshape(input_shape)

        input_data = np.repeat(input_data,num_samples,axis=0)

        mse = mean_squared_error(output_data,input_data).numpy().flatten()[0].astype(float)

        return mse


    def classifier_inference_lite(self,values,modelId):

        xInference = np.atleast_2d(np.array(values).astype(np.float32))


        if self.preload_models:
            interpreter = self.cnn_mlp_lite_model
        else:
            interpreter = tflite.Interpreter(model_path=self.basePath + "Models/MLP-Classifier/Lite/CNN-MLP-Lite.tflite")

        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Test model on random input data.
        input_shape = input_details[0]['shape']
        # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        input_data = xInference.reshape(input_shape).astype(np.float32)
        output_shape = output_details[0]['shape']


        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index']).flatten()
        
        classification = output_data[0].astype(float)

        return classification

    def model_gmm(self,values,modelId):

        xInference = np.atleast_2d(values).astype(np.float32)

        if self.preload_models:
            model = self.pca_gmm_model
        else:
            model = load(self.basePath + "Models/GMM/PCA-GMM.joblib")

        log_likelihood = model.score_samples(xInference)

        return log_likelihood.flatten()[0].astype(float)

    def model_gnb(self,values,modelId):

        xInference = np.atleast_2d(np.array(values).astype(np.float32))

        if self.preload_models:
            model = self.pca_gnb_model
        else:
            model = load(self.basePath + "Models/GNB/PCA-GNB.joblib")

        classification = model.predict_proba(xInference).flatten()[0]

        return classification


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)

