import os
import glob
import numpy as np

# modules for extracting the contours
from skimage import measure
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.morphology import binary_fill_holes

import sys
sys.path.append( "/mnt/alp/Users/Manuel/code/DeepStrain/")

# modules for generating the segmentation
from data import base_dataset
from tensorflow.keras.optimizers import Adam
from models import deep_strain_model

def Contours(mask, tissue_labels=[1,2,3]):
    contours = []
    for i in tissue_labels:
        mask_ = binary_fill_holes(mask==i)
        c = measure.find_contours(mask_,0.8)
        if len(c) == 0:  continue
        c = c[np.argmax([len(c) for c in c])]
        contours.append(c)
    return contours


def normalize(x, axis=(0,1,2)):
    # normalize per volume (x,y,z) frame
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True)
    return (x-mu)/(sd+1e-8)

def get_mask(V, netS):
    nx, ny, nz, nt = V.shape
    
    M = np.zeros((nx,ny,nz,nt))
    v = V.transpose((2,3,0,1)).reshape((-1,nx,ny)) # (nz*nt,nx,ny)
    v = normalize(v)
    m = netS(v[:,nx//2-64:nx//2+64,ny//2-64:ny//2+64,None])
    M[nx//2-64:nx//2+64,ny//2-64:ny//2+64] += np.argmax(m, -1).transpose((1,2,0)).reshape((128,128,nz,nt))
    print(M.shape)
    return M


class Options():
    
    def __init__(self):
        
        self.datadir = '../../../../datasets/ACDC'
        self.isTrain = False
        self.image_shape = (128,128,1)
        self.volume_shape = (128,128,16,1)
        self.nlabels = 4
        self.pretrained_models_netS  = '/home/mmorales/main_python/DeepStrain/pretrained_models/carson_Jan2021.h5'
        self.pretrained_models_netME = '/home/mmorales/main_python/DeepStrain/pretrained_models/carmen_Jan2021.h5'

def get_segmentation_model(): 

    opt = Options()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # use cpu

    model = deep_strain_model.DeepStrain(Adam, opt)
    netS  = model.get_netS()

    return netS

def get_mask(Array4D, netS):
    """Segmentation of LV and RV blood pools and LV myocardium. 

    Input
    -----

    Array4D : Array of shape n_ro, n_pe, n_slices, n_frames
    
    """

    n_ro, n_pe, n_slices, n_frames = Array4D.shape
    
    M = np.zeros((n_ro, n_pe, n_slices, n_frames), dtype=np.int16)

    Array4D = Array4D.transpose((2,3,0,1)).reshape((-1,n_ro,n_pe)) # (n_slices*n_frames,n_ro,ny)
    
    Array4D = normalize(Array4D)
    
    Array4D_ROI = netS(Array4D[:,n_ro//2-64:n_ro//2+64,n_pe//2-64:n_pe//2+64,None])

    Array4D_ROI[n_ro//2-64:n_ro//2+64,n_pe//2-64:n_pe//2+64] += np.argmax(Array4D_ROI, -1).transpose((1,2,0)).reshape((128,128,n_slices,n_frames))

    return Array4D_ROI

def mask_to_contours(mask, tissue_labels=[1,2,3]):
    """ Extract contours of 2D mask 
    
    Returns
    -------

    Contours : list of contours, where each contour is of shape (n_points, 2) for x, y coordinates.
    
    """
    Contours = []
    for i in tissue_labels:
        mask_ = binary_fill_holes(mask==i)
        c = measure.find_contours(mask_,0.8)
        if len(c) == 0:  continue
        c = c[np.argmax([len(c) for c in c])]
        Contours.append(c)
    
    return Contours

def PlotContours(ax, mask, tissue_labels=[1,2,3], 
                 contour_colors=['lime','magenta','red'],
                 contour_labels=['RV','LVM','LV'],
                 tolerance=0.1,
                 alpha=0.5,
                 linewidth=2,
                 legend=False):
    
    contours = Contours(mask, tissue_labels=tissue_labels)
    for i, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0],alpha=alpha, linewidth=linewidth, color=contour_colors[i],label=contour_labels[i])
    if legend:
        ax.legend(fontsize=26)  


n_ro = 144
n_pe = 144
n_slices = 1 
n_frames = 1

# replace this with input image
Array4D = np.ones((n_ro, n_pe, n_slices, n_frames))

netS = get_segmentation_model()

M = get_mask(Array4D, netS)
