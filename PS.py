import numpy as np

def compute_flux_power_spectrum(flux, dx=1):
    '''
    Flux:
        shape(number of spectra, length of spectra)
    '''
    F = flux
    N = F.shape[1]
    L = N * dx
    mean_flux = F.mean()
    delta_F = F / mean_flux - 1.0
    #k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    k = np.fft.fftfreq(N, d=dx)
    delta_fk = np.fft.fft(delta_F)
    Pk = (L / N**2) * np.abs(delta_fk)**2

    return Pk