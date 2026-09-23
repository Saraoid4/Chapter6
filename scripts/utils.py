import numpy as np
import torch
import torch.nn.functional as F

def standardize_frequency_shape(background_data, T=256, n_freq=256):
    x = torch.as_tensor(background_data, dtype=torch.float32)
    f, t = x.shape
    #x = x[:n_freq, :]
    x = x.unsqueeze(0).unsqueeze(0)  
    x = F.interpolate(x,size=(n_freq, t), mode="bilinear",align_corners=False)
    return x.squeeze(0).squeeze(0)

def standardize_data_shape_time(background_data, T=256, n_freq=256):
    x = torch.as_tensor(background_data, dtype=torch.float32)
    f, t = x.shape
    x = x[:n_freq, :]
    if t == T:
        return x
    x = x.unsqueeze(0).unsqueeze(0)  
    x = F.interpolate(x,size=(x.shape[-2], T), mode="bilinear",align_corners=False)
    return x.squeeze(0).squeeze(0)  

import numpy as np

def linear_to_db(x, ref=None, eps=1e-10, mode="power"):
    x = np.clip(x, eps, None)
    ref = x.max() if ref is None else ref
    factor = 10.0 if mode == "power" else 20.0
    return factor * np.log10(x / ref)