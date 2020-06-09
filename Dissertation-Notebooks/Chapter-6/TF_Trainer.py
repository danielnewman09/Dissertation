# !/home/dnewman/environments/tf-gpu/bin/python3
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
import tflite_runtime.interpreter as tflite
import numpy as np
from scipy import signal
import os

os.environ["PATH"] += os.pathsep + '/home/dnewman/environments/tf-gpu/bin'


class TF_Trainer(object):
    
    def __init__(
                self,
                feature,
                modelId,
                assetId,
                dataItemId,
                debug=False):
        
        self.assetId = assetId
        self.feature = feature
        self.dataItemId = dataItemId
        self.modelId = modelId


    def train_model(self,
                    model,
                    spindleSpeed,
                    isWarmUp,
                    startTime,
                    endTime,
                    batch_size,
                    max_epochs,
                    verbose=2,
                    sampling_interval=None
                    ):
        
        self.get_features(spindleSpeed,isWarmUp,startTime,endTime,sampling_interval)
        
        model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['mse'])
        loss = model.fit(
                    self.X_train,self.X_train,
                    validation_data=(self.X_test,self.X_test),
                    epochs=max_epochs,
                    batch_size=batch_size,
                    verbose=verbose)
        
        if not self.pretrain:
            output = self.save_model(model,spindleSpeed,isWarmUp)
        output = {'output':True}
        return output


    def save_model(self,model,spindleSpeed,isWarmUp):

        if isWarmUp:
            warmUpVal = 'TRUE'
        else:
            warmUpVal = 'FALSE'
            
        save_dir = '/home/dnewman/TF_Models/'
           
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        model.save(save_dir + 'model.h5',save_format='tf')
        
        with open(save_dir + 'model.h5', mode='rb') as file:
            data = file.read()
            
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            
            open(os.path.join(save_dir, 'model.tflite',),
                 "wb").write(tflite_model)
            with open(save_dir + 'model.tflite', mode='rb') as file:
                data_lite = file.read() 
                
            avgMeanLite,avgStdLite,varMeanLite,varStdLite = self.control_chart_data()
        
        except:
            print('Could not create TFLite Model')
        
        avgMeanFull,avgStdFull,varMeanFull,varStdFull = self.control_chart_data_full(model)
        
        return return_vals
    
    def control_chart_data(self):
        
        means,variances = self.fit_model_lite(self.X_test)
            
        avg_mean = np.mean(means)
        avg_std = np.std(means)

        var_mean = np.mean(variances)
        var_std = np.std(variances)
        
        return avg_mean,avg_std,var_mean,var_std   
    
    def fit_model_lite(self,data):
        means = np.zeros(data.shape[0])
        variances = np.zeros(data.shape[0])

        for i in range(self.X_test.shape[0]):
            mean,variance = self.lite_result(data[i,:,:].flatten())
            
            means[i] = mean
            variances[i] = variance
            
        return means,variances
    
    def lite_result(self,data):

        interpreter = tflite.Interpreter(model_path="/home/dnewman/TF_Models/model.tflite")
        interpreter.allocate_tensors()
        
    
        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test model on random input data.
        input_shape = input_details[0]['shape']
        # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        input_data = data.reshape(input_shape).astype(np.float32)

        num_samples = 1

        all_outputs = np.zeros((num_samples,input_shape[1],input_shape[2]))

        for i in range(num_samples):

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            output_data = interpreter.get_tensor(output_details[0]['index']).reshape(input_shape)

            all_outputs[i,:,:] = output_data

        input_data = np.repeat(input_data,num_samples,axis=0)


        mse = keras.metrics.mean_squared_error(all_outputs,input_data)
        means = np.mean(mse,axis=1)
        means = np.mean(means)

        variances = np.var(mse,axis=1).flatten()
        variances = np.var(variances)
        return means,variances

    def fit_model_full(self,data,model,num_samples=1):

        X_predict = np.repeat(data,num_samples,axis=0)
        predict = model.predict(X_predict)

        mse = keras.metrics.mean_squared_error(X_predict,predict)
        means = np.mean(mse,axis=1)
        variances = np.var(mse,axis=1).flatten()   
        
        return means,variances
        
    def control_chart_data_full(self,model):
        
        means,variances = self.fit_model_full(self.X_test,model)

        avg_mean = np.mean(means)
        avg_std = np.std(means)

        var_mean = np.mean(variances)
        var_std = np.std(variances)
        
        return avg_mean,avg_std,var_mean,var_std
        
    
