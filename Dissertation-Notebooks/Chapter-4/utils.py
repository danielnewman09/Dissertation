import numpy as np
from sklearn.mixture import BayesianGaussianMixture as GMM
from sklearn.naive_bayes import GaussianNB

import seaborn as sns
sns.set(style="ticks")

import pandas as pd

from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (15, 4)

def fit_GMM(data,num_components):
    gmm = GMM(n_components=num_components)
    gmm.fit(data)

    predicted_class = gmm.predict(data)
    num_classes = np.unique(gmm.predict(data)).shape[0]

    return gmm,predicted_class,num_classes

def fit_NB(data,predicted_class):
    
    clf = GaussianNB()
    clf.fit(data,predicted_class)

    return clf

def determine_probabilities(clf,data,predicted_class):

    num_classes = np.unique(predicted_class).shape[0]

    probability_scores = np.zeros((num_classes,num_classes))

    for i in range(num_classes):
        class_mask = np.argwhere(predicted_class == i)
        for j in range(num_classes):
            if data[class_mask.flatten(),:].size > 0:
                probability_scores[j,i] = clf.score(data[class_mask.flatten(),:],j * np.ones(class_mask.shape[0]))

    probability_scores = np.sort(probability_scores,axis=0)[::-1]

    return probability_scores


def lin_log_interp(fft_features):
    '''
    Scale the fft features from the logarithmic axis to be approximately on 
    the interval from 0 to 1
    '''
    
    # Minimum exponent we expect to see in the data
    minimum = -8
    
    # Maximum exponent we expect to see
    maximum = 0
    
    # Number of points to use for interpolation
    numpoints = 1000
    
    # Map the logarithmic x-axis to a linear y-axis
    x = np.logspace(minimum,maximum,numpoints)
    y = np.linspace(0,1,numpoints)

    # Return the interpolated valuess
    return np.interp(np.log10(fft_features),np.log10(x),y)

def show_statistical_fft(clf):

    fig = plt.figure(figsize=(15, 4))
    
    num_classes,num_features = clf.theta_.shape

    loc = clf.theta_
    scale = np.clip(clf.sigma_,1e-16,np.inf)
    allLabels = []
    df = pd.DataFrame()
    
    num_samples = 100
    
    for j in range(loc.shape[0]):
        tempDF = pd.DataFrame(np.random.normal(loc[j,:],scale[j,:],size=(num_samples,num_features)))
        labels = [clf.classes_[j] for k in range(num_samples)]
        allLabels = allLabels + labels
        df = df.append(tempDF,sort=False)
        
    frequencies = np.arange(0,2640,10.27961000)
    df['Label'] = allLabels
    thisDF = pd.DataFrame({"Label": np.repeat(df['Label'].values,257),
     "frequencies": np.tile(frequencies,df.shape[0]),
     "fftValue": np.array([np.array(row[:257]) for index, row in df.iterrows()]).astype(np.float64).flatten()})

    sns.lineplot(x="frequencies", y="fftValue",hue='Label',ci="sd",data=thisDF)
    plt.show()