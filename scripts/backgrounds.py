import numpy as np
import os
import glob

def get_array_from_npz(npz_obj, use_db=True):
        if "data" not in npz_obj:
            raise KeyError("Missing 'data' key in npz file")

        sdata = npz_obj["data"].item()
        arr = np.array(sdata.db if use_db else sdata.amp)

        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[:, :, 0]

        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {arr.shape}")

        return arr

def list_background_files(background_dir, pattern="*.npz"):
    if isinstance(background_dir, (list,tuple)):
        all_files =[]
        for d in background_dir:
            files = sorted(glob.glob(os.path.join(d, pattern)))
            all_files.extend(files)
        return sorted(all_files)

def split_backgrounds(files, test_frac=0.2, seed=21):
     rng = np.random.RandomState(seed)
     files = list(files)
     rng.shuffle(files)

     n_test = max(1, int(round(len(files)*test_frac)))
     test_files = sorted(files[:n_test])
     train_files = sorted(files[n_test:])

     return train_files, test_files

class BgCache:
    def __init__(self, use_db=False):
          self.use_db = use_db
          self._cache = {}
    def get(self, path):
        if path not in self._cache:
              d = np.load(path, allow_pickle=True)
              self._cache[path] = get_array_from_npz(d, use_db=self.use_db)
        return self._cache[path]

    