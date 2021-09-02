import numpy as np
import os

bp = os.path.join(os.environ['OV_DATA_BASE'], 'preprocessed', 'OV04', 'pod_2d')
fols = [fol for fol in os.listdir(bp) if fol.startswith('restaurations')]

for fol in fols:
    for case in os.listdir(os.path.join(bp, fol)):
        f = os.path.join(bp, fol, case)
        im = np.load(f).astype(np.float16)
        np.save(f, im)