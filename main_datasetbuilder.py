import sys
sys.path.append('/Users/selbouch/Desktop/Chapter6/scripts')
import json
import yaml
import numpy as np
import h5py

from backgrounds import list_background_files, split_backgrounds
from sampler import IndepSampler, LatinHypercubeSampler
from generator import Generator

fmin, fmax, dt = 400, 800, 0.004

def split_from_config(config):
    all_files = list_background_files(config["background_dir"])
    train_files, test_files = split_backgrounds(all_files, test_frac=config["test_frac"], seed=config["split_seed"])
    split = config["split"]
    if split == "train":
        return train_files 
    elif split == "test":
        return test_files
    else:
        raise ValueError("Unknown split in config")
    
def set_hcsampler_from_config(config):
    sampler = LatinHypercubeSampler.__new__(LatinHypercubeSampler)
    sampler.config = config
    sampler.param_specs = config["hypercube_params"]
    sampler.param_names = list(sampler.param_specs.keys())
    sampler.n_frb = config["n_frb"]
    sampler.n_background = config.get("n_background", 0)
    sampler.seed = config.get("split_seed",None)
    return sampler

def set_indep_sampler_from_config(config, seed=21):
    sampler = IndepSampler.__new__(IndepSampler)
    sampler.config = config
    sampler.param_specs = config[f"parameters_{sampler.config["split"]}"]
    sampler.n_frb = config["n_frb"]
    sampler.n_background = config.get("n_background",0)
    sampler.seed = config.get("split_seed",None)
    sampler.param_names = list(sampler.param_specs.keys())
    return sampler

def set_param_sampler_from_config(config):
    param_specs = config[f"parameters_{config["split"]}"]
    return IndepSampler(param_specs, n=config["n_frb"], seed=config.get("split_seed",None))

def build_positive_samples(config, files, rng):
    sampler = set_indep_sampler_from_config(config)
    params = sampler.sample_all()
    frbs = []
    for param in params:
        bg_path = files[rng.integers(0, len(files))]
        gen = Generator(background_dir=bg_path, param=param, positive=True, fmin=fmin, fmax=fmax, dt=dt, seed=rng.integers(0, 2**31-1))
        frbs.append({"data": gen.get_data(), "label":1, "background_file":bg_path, "param":param})
    return frbs

def build_negative_samples(config, files, rng):
    bg_params = {"dm":0.0, "snr":0, "width_s":0, "scat_s":0, "spec_id":0}
    n_backgrounds = config.get("n_background", 0)
    bgs = []
    for _ in range(n_backgrounds):
        bg_path = files[rng.integers(0, len(files))]
        gen = Generator(background_dir=bg_path, param=bg_params, positive=False, fmin=fmin, fmax=fmax, dt=dt, seed=rng.integers(0, 2**31-1))
        bgs.append({"data": gen.get_data(), "label":0, "background_file":bg_path, "param":bg_params})
    return bgs

#TODO: write ssamples to hdf5 files
def write_h5_file(samples, config):
    n = len(samples)
    target_shape = tuple(config["target_shape"])
    with h5py.file(config["output_file"], "w") as hf:
        data = hf.create_dataset("data", shape=(n, *target_shape), dtype=np.float32)
        labels = hf.create_dataset("label", shape=(n,), dtype=np.uint8)
        str_dtype = h5py.string_dtype()
        params = hf.create_data("parameters", shape=(n,), dtype=str_dtype)
        provenance_data = hf.create_data("bg_file", shape=(n,), dtype=str_dtype)
        for i, sam in enumerate(samples):
            data[i] = sam["data"].astype(np.float32)
            labels[i] = sam["label"].astype(np.uint8)
            provenance_data[i] = sam["background_file"]
            params[i] = json.write(sam["param"])


if __name__=='__main__':
    import matplotlib.pyplot as plt

    with open('scripts/config_train.yml', "r") as f:
        config_train = yaml.safe_load(f) 
    with open('scripts/config_test.yml', "r") as f:
        config_test = yaml.safe_load(f)
    rng = np.random.default_rng(42)
    full_train = split_from_config(config=config_train)
    sampler = set_indep_sampler_from_config(config=config_train)
    simus_pos = build_positive_samples(config_train, full_train, rng)
    simus_neg = build_negative_samples(config_train, full_train, rng)
    write_h5_file(simus_pos + simus_neg, config_train)