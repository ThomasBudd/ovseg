import numpy as np
import os
from tqdm import tqdm

prep = os.path.join(os.environ['OV_DATA_BASE'],
                    'preprocessed',
                    'kits21',
                    'kidney_full_refine')

lbp = os.path.join(prep, 'labels')
regp = os.path.join(prep, 'regions')


sens_list = []
for case in tqdm(os.listdir(lbp)):
    
    lb = np.load(os.path.join(lbp, case))
    reg = np.load(os.path.join(regp, case))
    sens_list.append(100 * np.sum(lb*reg)/np.sum(lb))
    

print([np.round(sens, 3) for sens in sens_list])
print(np.mean(sens_list))
print(np.median(sens_list))
