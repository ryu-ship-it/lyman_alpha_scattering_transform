import numpy as np
from scipy.spatial.distance import mahalanobis

def compute_mahalanobis_distance(arr1: np.ndarray, arr1_mean, arr2: np.ndarray, arr2_mean):

    X = np.vstack((arr1, arr2))
    cov = np.cov(X.T)
    inverse_cov = np.linalg.inv(cov)
    distance = mahalanobis(arr1_mean, arr2_mean, inverse_cov)

    return distance

#Normalizing power spectrum and coefficients
def normalizaiton(data):

    min = data.min(axis=0)
    max = data.max(axis=0)
    normalized_data = (data - min)/(max - min)
    
    return normalized_data
