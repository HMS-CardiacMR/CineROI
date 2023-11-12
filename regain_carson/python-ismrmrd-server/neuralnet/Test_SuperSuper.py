import skimage.color
import skimage.io
import skimage.metrics
import torchvision.utils
from PIL import Image

import matplotlib.pyplot as plt
from ConfigValid import *

import random
from typing import Tuple

import scipy.fftpack as sfft

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


    filenames = os.listdir(hr_dir)
    total_files = len(filenames)

    for index in range(total_files):
        lr_path = os.path.join(lr_dir, filenames[index])
        hr_path = os.path.join(hr_dir, filenames[index])
        hr_image = np.load(hr_path)
        lr_image = np.load(lr_path)


        hr_tensor = image2tensor(lr_image).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            sr_tensor = generator(hr_tensor.half())
            sr_tensor = sr_tensor.squeeze()
            img = torch.from_numpy(np.array(sr_tensor.to('cpu'), np.float32, copy=False))
            sr_image = np.array(img)


        if 'SAX' in hr_path:
            hr_image=np.rot90(hr_image,3)
            lr_image = np.rot90(lr_image,3)
            sr_image = np.rot90(sr_image,3)
        if '2ch' in hr_path:
            hr_image=np.transpose(hr_image)
            lr_image = np.transpose(lr_image)
            sr_image = np.transpose(sr_image)
        if '4ch' in hr_path:
            hr_image = np.flipud(hr_image)
            lr_image = np.flipud(lr_image)
            sr_image = np.flipud(sr_image)
        if 'LGE' in hr_path:
            hr_image = np.flipud(hr_image)
            lr_image = np.flipud(lr_image)
            sr_image = np.flipud(sr_image)
            hr_image = np.rot90(hr_image)
            lr_image = np.rot90(lr_image)
            sr_image = np.rot90(sr_image)
            hr_image = np.flipud(hr_image)
            lr_image = np.flipud(lr_image)
            sr_image = np.flipud(sr_image)



        BicubicImg = np.array(
            Image.fromarray(lr_image).resize((lr_image.shape[1], lr_image.shape[0]), Image.BICUBIC))

        lr_image = np.array(
            Image.fromarray(lr_image).resize((lr_image.shape[1], lr_image.shape[0]), Image.NEAREST))

        hr_complexKspace = sfft.fft2(hr_image)
        hr_complexKspace = PercentileRescaler(np.log(abs(sfft.fftshift(hr_complexKspace))+1))



        sr_complexKspace = sfft.fft2(sr_image)
        sr_complexKspace = PercentileRescaler(np.log(abs(sfft.fftshift(sr_complexKspace))+1))

        lr_complexKspace = sfft.fft2(lr_image)
        lr_complexKspace = PercentileRescaler(np.log(abs(sfft.fftshift(lr_complexKspace))+1))

#        catimage=np.concatenate((abs(hr_image),abs(lr_image),abs(sr_image)),1)
#        catimage=MinMaxRescaler(catimage)
#        catFFT = np.concatenate((abs(hr_complexKspace), abs(lr_complexKspace), abs(sr_complexKspace)), 1)

#        catcat=np.concatenate((catimage,catFFT),0)

        plt.imsave(sr_dir + "/HR/" + filenames[index] + "hr_image.tiff", abs(hr_image), cmap='gray')
        plt.imsave(sr_dir + "/LR/" + filenames[index] + "lr_image.tiff", abs(lr_image), cmap='gray')
        plt.imsave(sr_dir + "/SR/" + filenames[index] + "sr_image.tiff", abs(sr_image), cmap='gray')
#        plt.imsave(sr_dir + "/catcat/"+ filenames[index] + "cat_FFTimage.tiff", catcat, cmap='gray')
#        plt.imsave(sr_dir + "/cat/" + filenames[index] + "cat_image.tiff", catimage, cmap='gray')
        plt.imsave(sr_dir + "/Bicubic/" + filenames[index] + "BicubicImg.tiff", abs(BicubicImg), cmap='gray')


if __name__ == "__main__":
    main()
