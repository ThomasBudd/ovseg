import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pydicom

plt.close()
# get one dcm file
rawp = os.path.join(os.environ['OV_DATA_BASE'], 'raw_data', 'OV04_dcm')
patp = os.path.join(rawp, os.listdir(rawp)[0])
scanp = os.path.join(patp, os.listdir(patp)[0])

ds = pydicom.dcmread(os.path.join(scanp, os.listdir(scanp)[10]))

# image in HU
rec = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept

# remove outer region to show the volume of reconstruciton
outer = rec < -1024
mn, mx = -1024, rec.max()
im = (rec.clip(mn, mx) - mn)/(mx-mn)
im[outer] = 1
im = np.pad(im, (50, 50), constant_values=1)
plt.imshow(im, cmap='gray')
plt.axis('off')

# now draw the detector
n = 612
a = -1

def plot_rays(a=0, alpha=1.0):
    angles = np.linspace(-1*np.pi/8,np.pi/8,51)
    
    
    det = np.array([np.sin(angles) * n  , np.cos(angles) * n - n/2])
    
    rot = np.array([[np.cos(a), np.sin(a)], [-1*np.sin(a), np.cos(a)]])
    
    x_det, y_det = np.matmul(rot, det)
    
    x_det += n/2
    y_det += n/2
    
    plt.plot(x_det, y_det, 'deepskyblue', linewidth=3,alpha=alpha)
    
    x_source = -1*np.sin(a) * n/2 + n/2
    y_source = -1*np.cos(a) * n/2 + n/2
    
    for x,y in zip(x_det, y_det):
        plt.plot([x, x_source], [y, y_source], 'deepskyblue', linewidth=1, alpha=alpha)
    
plot_rays(a, 0.25)
plot_rays(0, 0.75)


x_source = -1*np.sin(a) * n/2 + n/2
y_source = -1*np.cos(a) * n/2 + n/2
style = "Simple, tail_width=0.5, head_width=4, head_length=8"
kw = dict(arrowstyle=style, color="k")
arrow = patches.FancyArrowPatch((n/2, 0),
                                (-1*np.sin(a) * n/2 + n/2, -1*np.cos(a) * n/2 + n/2),
                                connectionstyle="arc3,rad=-.3", **kw)
plt.gca().add_patch(arrow)
plt.savefig(os.path.join(os.environ['OV_DATA_BASE'], 'plots', 'CT_graphics.png'),
            bbox_inches='tight')

