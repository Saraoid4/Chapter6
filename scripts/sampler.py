import numpy as np
import yaml
from scipy.stats import qmc 

class Sampler:
    def __init__(self, config_path, seed=None):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.param_specs = self.config["hypercube_params"]
        self.param_names = list(self.param_specs.keys())
        self.n_frb = self.config["n_frb"]
        self.n_background = self.config.get("n_background",0)
        self.seed = seed

    def _map_unit_to_params(self, u, spec):
        low, high = spec["low"], spec["high"]
        if spec.get("log", False):
            log_low, log_high = np.log10(low), np.log10(high)
            return 10**(log_low + u*(log_high - log_low))
        return low + u*(high-low)
    
    def sample_all(self):
        d = len(self.param_names)
        sampler = qmc.LatinHypercube(d=d, seed=self.seed)
        unit_samples = sampler.random(n=self.n_frb)

        param_dicts = []
        for row in unit_samples:
            param = {name: self._map_unit_to_params(u, self.param_specs[name]) for name, u in zip(self.param_names, row)}
            param_dicts.append(param)
        return param_dicts