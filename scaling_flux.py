import numpy as np
from scipy import optimize

#results from arxiv:1904.01110, redshift 2-5, for HI
def compute_desired_mean_flux(z):

    tau_eff = 5.54e-3 * (1 + z)**3.182
    mean_flux = np.exp(-tau_eff)

    return mean_flux

def compute_flux_scale_factor(optical_depth, desired_mean_flux):

    f = lambda alpha: np.mean(np.exp(-alpha * optical_depth)) - desired_mean_flux
    alpha_root = optimize.brentq(f, 1e-6, 1e2)

    return alpha_root