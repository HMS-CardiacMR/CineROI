import skimage.color
import skimage.io
import skimage.metrics
import torchvision.utils
from PIL import Image

import matplotlib.pyplot as plt
from ConfigValid import *

import random
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.functional as F
from torch import Tensor


def image_quality_assessment(hr_image, sr_image, KeyName) -> Tuple[float, float]:
    #    npyimage = np.concatenate((hr_image, sr_image), 1)

    #    plt.imsave(sr_dir + KeyName + "cat.tiff", abs(npyimage), cmap='gray')

    hrmm = MinMaxRescaler(hr_image)
    sr_image = MinMaxRescaler(sr_image)
    psnr = skimage.metrics.peak_signal_noise_ratio(hr_image, sr_image)
    ssim = skimage.metrics.structural_similarity(hr_image, sr_image)
    nrmse = skimage.metrics.mean_squared_error(hr_image, sr_image)
    return psnr, ssim, nrmse


def image2tensor(image) -> Tensor:
    tensor = torch.from_numpy(np.array(image, np.float32, copy=False))

    return tensor


def tensor2image(tensor) -> np.ndarray:
    image = F.to_pil_image(tensor)
    return image


def MinMaxRescaler(Arr):
    minval = np.min(Arr)
    maxval = np.max(Arr)

    Arr = (Arr - minval) / (maxval - minval)

    return Arr
def PercentileRescaler(Arr):
    minval=np.percentile(Arr, 3, axis=None, out=None, overwrite_input=False, interpolation='linear', keepdims=False)
    maxval=np.percentile(Arr, 97, axis=None, out=None, overwrite_input=False, interpolation='linear', keepdims=False)

    if minval==maxval:
        print("Zero Detected")
    Arr=(Arr-minval)/(maxval-minval)
    Arr=np.clip(Arr, 0.0, 1.0)
    return Arr

def main() -> None:
    generator.half()
    generator.eval()


    filenames = os.listdir(lr_dir)
    total_files = len(filenames)

    FolderList={"HR","LR","SR","catcat","cat","HR_numpy","LR_numpy","SR_numpy"}

    for index in range(total_files):
        print (filenames[index])


        target = str(filenames[index]).lower()
        s = target.split('_')
        Pid=s[0]
        SequenceId = s[1]

        for folderId in FolderList:
            OutFolder=os.path.join(sr_dir+'/'+folderId,Pid+'/'+SequenceId)
            if not os.path.exists(OutFolder):
                os.makedirs(OutFolder)



        lr_path = os.path.join(lr_dir, filenames[index])
        hr_path = os.path.join(hr_dir, filenames[index])
        hr_image = np.load(hr_path)
        lr_image = np.load(lr_path)
        hr_image= PercentileRescaler(hr_image)
        lr_image = PercentileRescaler(lr_image)
        hr_tensor = image2tensor(lr_image).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            sr_tensor = generator(hr_tensor.half())
            sr_tensor = sr_tensor.squeeze()
            img = torch.from_numpy(np.array(sr_tensor.to('cpu'), np.float32, copy=False))
            sr_image = np.array(img)

if __name__ == "__main__":
    main()
