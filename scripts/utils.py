import numpy as np
import torch
import torch.nn.functional as F
from generator import FRBConstants

T = FRBConstants.DEFAULT_CROP_SIZE
N_FREQ = FRBConstants.BAND_LIMIT_CHANNELS

def standardize_data_shape_time(background_data, T=256, n_freq=256):
    x = torch.as_tensor(background_data, dtype=torch.float32)
    f, t = x.shape

    x = x[:n_freq, :]
    if t == T:
        return x
    x = x.unsqueeze(0).unsqueeze(0)  
    x = F.interpolate(x,size=(x.shape[-2], T), mode="bilinear",align_corners=False)
    return x.squeeze(0).squeeze(0)  