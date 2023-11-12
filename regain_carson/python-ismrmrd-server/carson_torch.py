import torch 
import numpy as np
from skimage import measure
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.morphology import binary_fill_holes

def Contours(mask, tissue_labels=[1,2,3]):
    contours = []
    for i in tissue_labels:
        mask_ = binary_fill_holes(mask==i)
        c = measure.find_contours(mask_,0.8)
        if len(c) != 0:
            c = c[np.argmax([len(c) for c in c])]
            contours.append(c)
    return contours

def normalize(x):
    mu = x.mean(keepdims=True)
    sd = x.std(keepdims=True)
    return (x-mu)/(sd+1e-8)

def get_mask(image, netS, device):
    """Segmentation of LV and RV blood pools and LV myocardium. 
    Input
    -----
    Array4D : Array of shape n_ro, n_pe, n_slices, n_frames
    """
    n_ro, n_pe = image.shape
 
    image_ROI = np.zeros((n_ro, n_pe), dtype=np.int16)

    with torch.no_grad():

        im = normalize(image)[None, n_ro//2-64:n_ro//2+64,n_pe//2-64:n_pe//2+64, None].copy()
        
        image_tensor = torch.from_numpy(np.array(im, np.float32, copy=False)).to(device)

        M = netS(image_tensor).to('cpu').detach().numpy()

        image_ROI[n_ro//2-64:n_ro//2+64,n_pe//2-64:n_pe//2+64] += np.argmax(M, -1).squeeze()

        del im, image_tensor  

    return image_ROI

def mask_to_contours(mask, tissue_labels=[1,2,3]):
    """ Extract contours of 2D mask 
    Returns
    -------
    Contours : list of contours, where each contour is of shape (n_points, 2) for x, y coordinates.
    """
    Contours = []
    mask = np.squeeze(mask)

    for i in tissue_labels:
        mask_ = binary_fill_holes(mask==i)
        c = measure.find_contours(mask_,0.8)
        if len(c) != 0:
            c = max(c, key=len)
            Contours.append(c)
    
    return Contours

