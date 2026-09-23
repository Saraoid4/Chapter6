import numpy as np
import sys
sys.path.append('/Users/selbouch/Desktop/Chapter6/injectfrb/injectfrb-backup/injectfrb')
from simulate_frb import gen_simulated_frb
import spectra
import os
import glob
from copy import deepcopy
import random
from backgrounds import BgCache, get_array_from_npz
from utils import standardize_data_shape_time

def loguniform(low,high,size=None):
        return np.exp(np.random.uniform(low,high,size))

class FRBConstants:
    DEFAULT_FLUENCE = 1
    DEFAULT_SPEC_IND = 0.0
    FREQ_THRESHOLD = 200 
    TIME_EXTENSION_FACTOR = 100 
    DEFAULT_CROP_SIZE = 256
    DEFAULT_CENTER_INDEX = DEFAULT_CROP_SIZE * TIME_EXTENSION_FACTOR // 2
    BAND_LIMIT_CHANNELS = 256


class FRB:
    def __init__(self, param, positive, fmin=400, fmax=800, dt=0.01):
        self.param = param
        self.dm = param["dm"]
        self.snr = param["snr"]
        self.width_s = param["width_s"]
        self.scat_s = param["scat_s"]
        self.spec_id = param["spec_id"]
        self.positive = positive
        self.dt = dt
        self.n_chann = FRBConstants.BAND_LIMIT_CHANNELS
        self.freq_rang = (fmax,fmin)
        self.freq_list = np.linspace(fmax, fmin, self.n_chann)
        self.freq_ref = fmax
        self.sigma = None
        self.simulated_final = None
    

    def _set_reference_frequency(self):
        max_attempts = 1000  
        attempts = 0

        while attempts < max_attempts:
            f_ind = int(np.random.randint(0, FRBConstants.BAND_LIMIT_CHANNELS - 1))
            freq_ref = self.freq_list[f_ind]
            self.f_ind = f_ind

            if abs(freq_ref) > FRBConstants.FREQ_THRESHOLD:
                return freq_ref
            attempts += 1

        abs_freqs = np.abs(self.freq_list)
        self.f_ind = int(np.argmax(abs_freqs))
        return self.freq_list[self.f_ind]
    
    def _calculate_noise_statistics(self):
        if self.bg_data is None or self.freq_list is None:
            raise ValueError("Background data must be loaded before calculating noise statistics")

        background_spectra = spectra.Spectra(freqs=self.freq_list,dt=self.dt,data=deepcopy(self.bg_data),starttime=0,dm=self.dm)
        background_spectra.dedisperse(self.dm, ref_freq=self.freq_ref)
        masked_arr = np.ma.masked_where(background_spectra.data == 0, background_spectra.data)
        profile = np.ma.mean(masked_arr, axis=0)
        self.sigma = np.mean(np.abs(profile - np.ma.mean(profile)))

    def _estimate_pulse_peak(self, integrated):
        max_intergrated = np.max(integrated)
        scale_factor = self.snr*self.sigma/max_intergrated
        return max_intergrated, scale_factor

    def get_data(self):
        return self.simulated_final

CACHE = BgCache(use_db=False)
class Generator(FRB):
    def __init__(self, background_dir, param, positive, fmin, fmax, dt, seed=None):
        super().__init__(param, positive, fmin, fmax, dt)
        self._rng = np.random.RandomState(seed)
        self._load_background_nenufar(background_dir, file_idx=0)
        if positive:
            self._calculate_noise_statistics()
            self.simulated_final = self._create_simulation()
        else:
            self.simulated_final = standardize_data_shape_time(self.bg_data).numpy()



    def _load_background_nenufar(self, background_dir, file_idx=0, rebin=True):
        if os.path.isdir(background_dir):
            file_pattern = "*.npz"
            files = sorted(glob.glob(os.path.join(background_dir, file_pattern)))
            path = files[file_idx]
        else:
            path = background_dir
        background = deepcopy(CACHE.get(path))
        self.source_file = path
        if rebin:
            rebin_factor = int(background.shape[1]//256)
            rebinned_arr = background.reshape(background.shape[0], rebin_factor, 256).mean(axis=1)
        else:
            rebinned_arr = background
        self.bg_data = np.swapaxes(rebinned_arr, 0, 1) 
        
    
    def _create_simulation(self):
        if self.bg_data is None:
            raise ValueError("Background data must be loaded before creating simulation")
        k_dm = 1e3/0.241
        sweep_s = k_dm*self.dm*(self.freq_rang[1]**-2 - self.freq_rang[0]**-2)
        total_observation = int((sweep_s + 10*self.scat_s + 0.25)/self.dt)
        #n_time_long = max(total_observation, self.bg_data.shape[1] * FRBConstants.TIME_EXTENSION_FACTOR)
        n_time_long = max(total_observation*2, self.bg_data.shape[1] * FRBConstants.TIME_EXTENSION_FACTOR)+ FRBConstants.DEFAULT_CROP_SIZE
        large_background = np.zeros((self.bg_data.shape[0], n_time_long))
        frb = gen_simulated_frb(dm=self.dm, fluence = FRBConstants.DEFAULT_FLUENCE, width=self.width_s, NFREQ = large_background.shape[0], NTIME = n_time_long,
                                scat_tau_ref=self.scat_s, scintillate=True, spec_ind=self.spec_id, background_noise=large_background, freq=self.freq_rang, FREQ_REF=self.freq_ref,
                                delta_t = self.dt, conv_dmsmear=False)
        frb_spectra = spectra.Spectra(freqs=self.freq_list, dt=self.dt, data=deepcopy(frb[0]), starttime=0, dm=self.dm)
        frb_spectra.dedisperse(self.dm, ref_freq=self.freq_ref)
        integrated = np.mean(frb_spectra.data, axis=0)
        max_integrated, scale_factor = self._estimate_pulse_peak(integrated)

        #self.index_start = random.randint(-100, 100)
        #new_center = n_time_long//2 - self.index_start
        center_factor = loguniform(-3,0)
        #new_center = n_time_long//2 - self.index_start
        new_center = int(n_time_long//2 + center_factor*(sweep_s/self.dt))
        crop_start = new_center - FRBConstants.DEFAULT_CROP_SIZE // 2
        crop_end = new_center + FRBConstants.DEFAULT_CROP_SIZE // 2
        frb_cropped = frb[0][:, crop_start:crop_end]
        self.burst = frb_cropped

        #Todo standardize the shapes for all background: add in utils
        x = standardize_data_shape_time(self.bg_data)
        final = scale_factor*frb_cropped + deepcopy(x.numpy())

        return final


if __name__=="__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    params = {"dm":600, "snr":20, "width_s":1e-3, "scat_s":0, "spec_id":0}
    test_frb_generator = Generator(param=params, positive=True, background_dir="./data_npz/snippetsHF_4768x256_BA", fmin=400, fmax=800, dt=0.01)
    #nenufar_bg = test_frb_generator._load_background_nenufar(background_dir="./data_npz/snippetsHF_4768x256_BA", file_idx=0)
    fin = test_frb_generator.simulated_final
        
        

        
