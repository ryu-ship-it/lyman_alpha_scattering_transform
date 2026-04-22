import numpy as np
import glob
import h5py
from collections import Counter

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
            try:
                spectra_file = h5py.File(file, 'r')
                for key in spectra_file['OpticalDepth'].keys():
                    spectra = np.array(spectra_file[f'OpticalDepth/{key}/Hydrogen_HI/Optical_depths'][:])
                    spectra_data.append(spectra)
                print('File loaded:', file)

            #skip corrupted datafile 
            except OSError:
                print('File skipped/corrupted:', file)
        try:
            optical_depth = np.array(spectra_data)
        except ValueError:
            #skip spectra with different length
            print('Dataset contains spectra with inconsistent length')
            lengths = [len(spectra) for spectra in spectra_data]
            length_of_spectra = Counter(lengths).most_common()[0][0]
            spectra_data_cleaned = [spectra for spectra in spectra_data if len(spectra) == length_of_spectra]
            optical_depth = np.array(spectra_data_cleaned)

        self.optical_depth = optical_depth
        return self.optical_depth
    
    def flux(self):
        return np.exp(-self.optical_depth)
