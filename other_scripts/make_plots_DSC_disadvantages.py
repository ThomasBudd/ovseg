import numpy as np
import matplotlib.pyplot as plt
import os

plotp = os.path.join(os.environ['OV_DATA_BASE'],
                     'plots')

n = 200
r = 0.5

gt = (np.sum(np.stack(np.meshgrid(*(2*[np.linspace(-1,1,n)])))**2,0) <= r).astype(float)
p1 = gt.copy()
p1[:n//2] = 0

plt.subplot(1, 3, 1)
plt.imshow(p1, cmap='gray')
plt.contour(gt, colors='r', linewidths=0.5)
plt.axis('off')

k = int(np.sqrt(np.sum(gt)/2))//2

p2 = np.zeros_like(gt)
p2[n//2-k:n//2+k,n//2-k:n//2+k] = 1

plt.subplot(1, 3, 2)
plt.imshow(p2, cmap='gray')
plt.contour(gt, colors='r', linewidths=0.5)
plt.axis('off')

p3 = np.zeros_like(gt)
k = 10
o = np.ones(k)
z = np.zeros(k)
for i in range(n):
    if (i % (2 * k)) < k:
        v = np.concatenate([o, z])
    else:
        v = np.concatenate([z, o])
    
    c = n // len(v)
    v = np.concatenate(c*[v])
    p3[i, v > 0] = 1
p3 = p3*gt

plt.subplot(1, 3, 3)
plt.imshow(p3, cmap='gray')
plt.contour(gt, colors='r', linewidths=0.5)
plt.axis('off')

plt.savefig(os.path.join(plotp, 'DSC_shape_unawareness.png'), bbox_inches='tight')
plt.close()

# %%

def DSC(s,p):
    return 200 * np.sum(s*p) / np.sum(s+p)

n = 512

r1 = 10
r2 = 100

x1, y1, x2, y2, x3, y3 = 150, 150, 450, 450, 150, 450

mg = np.stack(np.meshgrid(np.arange(n),np.arange(n)))
c11 = (np.sum((mg - np.array([x1,y1]).reshape((2,1,1))) ** 2, 0)**0.5 < r1).astype(float)
c12 = (np.sum((mg - np.array([x1,y1]).reshape((2,1,1))) ** 2, 0)**0.5 < r2).astype(float)
c21 = (np.sum((mg - np.array([x2,y2]).reshape((2,1,1))) ** 2, 0)**0.5 < r1).astype(float)
c31 = (np.sum((mg - np.array([x3,y3]).reshape((2,1,1))) ** 2, 0)**0.5 < r1).astype(float)

s1 = c11 + c21
s2 = c12 + c21
p1 = c11 + c31
p2 = c12 + c31
plt.subplot(1, 2, 1)
plt.imshow(p1, cmap='gray')
plt.contour(s1, colors='r', linewidths=0.5)
plt.axis('off')
plt.title('DSC={:.2f}'.format(DSC(s1,p1)))

plt.subplot(1, 2, 2)
plt.imshow(p2, cmap='gray')
plt.contour(s2, colors='r', linewidths=0.5)
plt.axis('off')
plt.title('DSC={:.2f}'.format(DSC(s2,p2)))

plt.savefig(os.path.join(plotp, 'DSC_volume_dependency.png'), bbox_inches='tight')
plt.close()
