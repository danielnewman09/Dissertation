import numpy as np
from run_models import RunModels
from time import process_time 
#import pandas as pd


modelIds = ['PCA-GMM','AE','CNN-AE','CNN-AE-Lite','PCA-GNB','MLP','CNN-MLP','CNN-MLP-Lite']
data_labels = ['Train_Healthy','Valid_Healthy','Train_Unhealthy','Valid_Unhealthy']

INFERENCE_LOCATION = 'Desktop'
MODEL_STATUS = 'Loaded'
# BASE_PATH = '/home/dnewman/Git/Edge-Analytics-IoT-Framework/'
BASE_PATH = '/home/dnewman/Documents/Github/Edge-Analytics-IoT-Framework/'

if INFERENCE_LOCATION == 'Desktop':
        import tensorflow as tf
        gpus= tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)


if MODEL_STATUS == 'Loaded':
    preload_models = True
else:
    preload_models = False

thisModel = RunModels(preload_models,BASE_PATH,desktop=True)

for k in range(4):

    testData = np.genfromtxt(data_labels[k] + '.csv',delimiter=',')
    latency = np.zeros((testData.shape[0],len(modelIds)))
    inferenceVals = np.zeros((testData.shape[0],len(modelIds)))

        
    for i in range(testData.shape[0]):

        values = testData[i,:].flatten()

        if i % 10 == 0:
            print('Processing Sample {}'.format(i))

        for j in range(len(modelIds)):
            modelId = modelIds[j]

            start_time = process_time()
            value = 0.

            if j == 0:
                #pass
                value = thisModel.model_gmm(values,modelId)
            elif j == 1 or  j == 2:
                pass
                #value = thisModel.model_inference_full(values,modelId)
            elif j == 3:
                #pass
                value = thisModel.model_inference_lite(values,modelId)
            elif j == 4:   
                #pass
                value = thisModel.model_gnb(values,modelId) 
            elif j == 5 or j == 6:
                pass
                #value = thisModel.classifier_inference_full(values,modelId)
            elif j == 7:
                #pass
                value = thisModel.classifier_inference_lite(values,modelId)

            end_time = process_time()

            thisLatency = end_time - start_time

            latency[i,j] = thisLatency
            inferenceVals[i,j] = value


    #myDF = pd.DataFrame(data=latency,columns=modelIds)
    #myDF.to_csv('Results/' + INFERENCE_LOCATION + '_' + MODEL_STATUS + '_' + data_labels[k] + '.csv')

    #myDF = pd.DataFrame(data=inferenceVals,columns=modelIds)
    #myDF.to_csv('Results/' + INFERENCE_LOCATION + '_' + MODEL_STATUS + '_' + data_labels[k] +  '_values.csv')

    modelHeader = ','.join(Id for Id in modelIds)
    np.savetxt('Results/' + INFERENCE_LOCATION + '1_' + MODEL_STATUS + '_' + data_labels[k] + '.csv',latency,delimiter=',',header=modelHeader)
    np.savetxt('Results/' + INFERENCE_LOCATION + '1_' + MODEL_STATUS + '_' + data_labels[k] + '_values.csv',inferenceVals,delimiter=',',header=modelHeader)





            

            
