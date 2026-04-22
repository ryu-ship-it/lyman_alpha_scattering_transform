import numpy as np
import glob
import h5py

class load_data:
    '''
    load psectra certain datafiles from path by redshift
    '''
    def __init__(self, redshift):
        self.redshift = redshift

    def optical_depth(self):
        spectra_dir = '/disks/cosmodm/maria/lyman_alpha'
        spectra_list = glob.glob(f'{spectra_dir}/*z{self.redshift}*.hdf5')
        spectra_data = []
        for file in spectra_list:
            spectra_file = h5py.File(file, 'r')
            for key in spectra_file['OpticalDepth'].keys():
                spectra = np.array(spectra_file[f'OpticalDepth/{key}/Hydrogen_HI/Optical_depths'][:])
                spectra_data.append(spectra)
        n_spectra_per_file = len(spectra_file['OpticalDepth'].keys())
        numof_files = len(spectra_list)
        numof_spectra = n_spectra_per_file * numof_files
        optical_depth = np.array(spectra_data).reshape(numof_spectra, -1)
        self.optical_depth = optical_depth
        return self.optical_depth
    
    def flux(self):
        return np.exp(-self.optical_depth)
