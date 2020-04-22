
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM

from joblib import dump, load

# mse = keras.metrics.mean_squared_error(all_outputs,input_data)

train_featuresDF = pd.read_csv('data/featuresDF_train.csv')
train_stats = pd.read_csv('data/stats_train.csv')
train_vibState = pd.read_csv('data/vibState_train.csv')

valid_featuresDF = pd.read_csv('data/featuresDF_valid.csv')
valid_stats = pd.read_csv('data/stats_valid.csv')
valid_vibState = pd.read_csv('data/vibState_valid.csv')

enc = OrdinalEncoder()
X_train = train_featuresDF.values[:,1:1025].astype(np.float32)
Y_train = enc.fit_transform(train_vibState.values[:,1][...,np.newaxis])

X_valid = valid_featuresDF.values[:,1:1025].astype(np.float32)
Y_valid = enc.transform(valid_vibState.values[:,1][...,np.newaxis])

X_train_healthy = X_train[np.argwhere(Y_train.flatten()==1).flatten(),:]
Y_train_healthy = Y_train[np.argwhere(Y_train.flatten()==1).flatten()]
X_train_unhealthy = X_train[np.argwhere(Y_train.flatten()==0).flatten(),:]
Y_train_unhealthy = Y_train[np.argwhere(Y_train.flatten()==0).flatten()]

X_valid_healthy = X_valid[np.argwhere(Y_valid.flatten()==1).flatten(),:]
Y_valid_healthy = Y_valid[np.argwhere(Y_valid.flatten()==1).flatten()]
X_valid_unhealthy = X_valid[np.argwhere(Y_valid.flatten()==0).flatten(),:]
Y_valid_unhealthy = Y_valid[np.argwhere(Y_valid.flatten()==0).flatten()]
np.random.shuffle(X_train_healthy)
np.random.shuffle(X_valid_healthy)
np.random.shuffle(X_train_unhealthy)
np.random.shuffle(X_valid_unhealthy)

X = np.dstack((X_train_healthy,X_valid_healthy,X_train_unhealthy,X_valid_unhealthy))

num_samples = np.array([X_train_healthy.shape[0],
          X_valid_healthy.shape[0],
          X_train_unhealthy.shape[0],
          X_valid_unhealthy.shape[0]])

columns = ['Healthy Train','Healthy Valid','Unhealthy Train','Unhealthy Valid']

estimators = [('reduce_dim', PCA(n_components=64)), ('gmm', GaussianMixture())]
# estimators = [('reduce_dim', KernelPCA(n_components=32,kernel='rbf')), ('gmm', GaussianMixture())]

pipe = Pipeline(estimators)
pipe.fit(X_train_healthy)

scores_gmm = np.zeros((max_samples,4))

for i in range(len(columns)):
    scores_gmm[:,i] = pipe.score_samples(X[...,i])

max_samples = np.amax(num_samples)

dump(pipe, 'data/pca_gmm_bbb.joblib') 