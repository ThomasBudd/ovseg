import numpy as np
from scipy.special import erf
from scipy.stats import ttest_ind
n = 30
t_raw = [30, 60, 90, 165, 70, 45, 145, 120, 80, 70, 75, 75, 80, 50, 75, 65, 60, 45,
         30, 45, 40, 50, 50, 55, 30, 45, 70, 100, 40, 100, 65, 70, 40, 45, 70, 
         80, 65, 100, 120, 60, 120, 85, 100, 30, 30, 25, 40, 15, 15, 40, 60, 20, 25, 10]

t_raw_new = [10+10, 15+30, 22+37, 30, 5+35, 30+10, 30+10, 15+35, 5+45, 0+40, 
             5+30, 25+0, 10+25, 15+15, 10+30, 15+30, 10+40, 15+0, 20+20, 25+25,
             25+35, 10+20, 25+45, 15+40, 30+0, 10+25, 15+30, 40+14, 20+35, 20+60,
             15+60, 10+35, 40+60, 0+30, 10+1.5, 20+10, 30, 10+15, 25+15, 15, 15, 
             10+30, 15+45, 20, 15, 25, 10, 10]

t_raw = np.array(t_raw)

mu0 = np.mean(t_raw_new)
sigma = np.std(t_raw_new)
print(mu0, sigma)

mu = np.linspace(0, mu0, 100)

t = (mu0 - mu) / sigma / np.sqrt(n)

phi = (1 - 2*erf(t))

# %%
t12 = np.random.choice(t_raw_new, size=2*n, replace=True)

t1 = t12[:n]
t2 = t12[n:]
# %%
t3 = t2 - 11

print(ttest_ind(t1, t3))
