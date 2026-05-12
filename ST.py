import numpy as np
import kymatio
from kymatio.numpy import Scattering1D

def spectra_scattering(data, J, Q=1):
    
    T = data.shape[-1]
    scattering = Scattering1D(J, T, Q)
    SC = scattering(data)
    meta = scattering.meta()
    return SC, meta

def group_by_order(coefficients: np.ndarray, meta: np.ndarray, order: int=0):
    '''
    group coefficients and corresponding keys from the meta data by order
    order:
        maximum 2
    '''
    idx = np.where(meta['order'] == order)[0].astype(int)
    keys = np.array(meta['key'], dtype=object)
    coefs_keys = keys[idx]
    coefs_order = coefficients[:, idx]
    
    return coefs_order, coefs_keys

def coefficients_normalization(Sn, Sn_lastorder, Sn_keys, order=1):
    '''
    normalize coefficients using Sn/Sn-1
    '''
    if order == 1:
        S1_normalized = Sn / Sn_lastorder
        return S1_normalized
        
    elif order == 2:
        j1_j2 = np.array(list(Sn_keys.flatten()))
        S2_normalized = {}
        S2_normalized['S2'] = {}
        S2_keys = []
        #select second order coefficients and keys by j1
        for j in np.unique(j1_j2[:, 0]):
            idx = np.where(j1_j2[:, 0] == j)[0]
            #S2_z4_grouped[f'{j}'] = S2_z4[:, idx]
            normalized_coefs = Sn[:, idx]/Sn_lastorder[:, j][:, np.newaxis]
            S2_normalized['S2'][f'{j}'] = normalized_coefs
            S2_keys.append(Sn_keys[idx])

        S2_normalized['j1_j2'] = S2_keys

        return S2_normalized
    
def get_normalized_S1_S2(coefficients, meta):
    '''
    Get the grouped and normalized S1 and S2
    '''
    
    S1, S1_keys = group_by_order(coefficients, meta, 1)
    S2, S2_keys = group_by_order(coefficients, meta, 2)
    S0, _ = group_by_order(coefficients, meta, 0)

    S1_normalized = {}
    S2_normalized = {}

    S1_normalized['S1'] = coefficients_normalization(S1, S0, S1_keys, 1)
    S1_normalized['j1'] = S1_keys

    S2_coefficients = coefficients_normalization(S2, S1, S2_keys, 2)
    S2_normalized['S2'] = S2_coefficients['S2']
    S2_normalized['j1_j2'] = S2_coefficients['j1_j2']

    return S1_normalized, S2_normalized