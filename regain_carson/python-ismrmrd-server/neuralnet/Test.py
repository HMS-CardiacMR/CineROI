
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

    hrmm= MinMaxRescaler(hr_image)
    sr_image= MinMaxRescaler(sr_image)
    psnr = skimage.metrics.peak_signal_noise_ratio(hr_image,sr_image)
    ssim = skimage.metrics.structural_similarity(hr_image, sr_image)
    nrmse=skimage.metrics.mean_squared_error(hr_image, sr_image)
    return psnr, ssim, nrmse




def image2tensor(image) -> Tensor:

    tensor = torch.from_numpy(np.array(image, np.float32, copy=False))


    return tensor


def tensor2image(tensor) -> np.ndarray:
    image = F.to_pil_image(tensor)
    return image

def MinMaxRescaler(Arr):
    minval =np.min(Arr)
    maxval =np.max(Arr)

    Arr =(Arr -minval ) /(maxval -minval)

    return Arr


def main() -> None:
    generator.half()
    generator.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    total_Networkpsnr = []
    total_Networkssim = []
    total_Networknrmse = []
    total_Bicubicpsnr = []
    total_Bicubicssim = []
    total_Bicubicnrmse = []
    total_Zerpadpsnr = []
    total_Zerpadssim = []
    total_Zerpadnrmse = []

    filenames = os.listdir(lr_dir)
    total_files = len(filenames)

    for index in range(total_files):
        lr_path = os.path.join(lr_dir, filenames[index])
        sr_path = os.path.join(sr_dir, filenames[index])
        hr_path = os.path.join(hr_dir, filenames[index])
        pad_path = os.path.join(pad_dir, filenames[index])
        # LR to SR.

        lr = np.load(lr_path)
        lr_tensor = image2tensor(lr).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            sr_tensor = generator(lr_tensor.half())
            sr_tensor =sr_tensor.squeeze()
            img = torch.from_numpy(np.array(sr_tensor.to('cpu'), np.float32, copy=False))
            sr_image = np.array(img)

        hr_image = np.load(hr_path)
        lr_image = np.load(lr_path)
#        pad_image = np.load(pad_path)
        ResizeImg = np.array(
                Image.fromarray(lr_image).resize((hr_image.shape[1], hr_image.shape[0]), Image.NEAREST))

        BicubicImg = np.array(
                Image.fromarray(lr_image).resize((hr_image.shape[1], hr_image.shape[0]), Image.BICUBIC))


#        npyimage = np.concatenate((hr_image, ResizeImg, BicubicImg,  pad_image, sr_image), 1)
        
#        plt.imsave(sr_dir+"/Cat/"+ filenames[index]+ "cat.tiff", abs(npyimage), cmap='gray')

        plt.imsave(sr_dir+"/LR/"+ filenames[index]+ "lr_image.tiff", abs(ResizeImg), cmap='gray')

        plt.imsave(sr_dir+"/HR/"+ filenames[index]+ "hr_image.tiff", abs(hr_image), cmap='gray')

        plt.imsave(sr_dir+"/SR/"+ filenames[index]+ "sr_image.tiff", abs(sr_image), cmap='gray')

        plt.imsave(sr_dir+"/Bicubic/"+ filenames[index]+ "BicubicImg.tiff", abs(BicubicImg), cmap='gray')

#        plt.imsave(sr_dir+"/Zeropadding/"+ filenames[index]+ "pad_image.tiff", abs(pad_image), cmap='gray')


    # Test PSNR and SSIM.
        print(f"Test `{lr_path}`.")
        Networkpsnr, Networkssim, Networknrmse = image_quality_assessment(hr_image , sr_image,'Network')

        Bicubicpsnr, Bicubicssim, Bicubicnrmse = image_quality_assessment(hr_image, BicubicImg,'Bicubic')

#        Zerpadpsnr, Zerpadssim, Zerpadnrmse = image_quality_assessment(hr_image, pad_image,'Zerpad')

        total_Networkpsnr.append(Networkpsnr)
        total_Networkssim.append(Networkssim)
        total_Networknrmse.append(Networknrmse)

        total_Bicubicpsnr.append(Bicubicpsnr)
        total_Bicubicssim.append(Bicubicssim)
        total_Bicubicnrmse.append(Bicubicnrmse)

#        total_Zerpadpsnr.append(Zerpadpsnr)
#        total_Zerpadssim.append(Zerpadssim)
#        total_Zerpadnrmse.append(Zerpadnrmse)

    print(f"total_Networkpsnr: ave {np.average(total_Networkpsnr)}, std {np.std(total_Networkpsnr)}\n"
          f"total_Networkssim:  ave {np.average(total_Networkssim)}, std {np.std(total_Networkssim)}\n"
          f"total_Networknrmse:  ave {np.average(total_Networknrmse)}, std {np.std(total_Networknrmse)}\n.")

    print(f"total_Bicubicpsnr: ave {np.average(total_Bicubicpsnr)}, std {np.std(total_Bicubicpsnr)}\n"
          f"total_Bicubicssim:  ave {np.average(total_Bicubicssim)}, std {np.std(total_Bicubicssim)}."
          f"total_Bicubicnrmse:  ave {np.average(total_Bicubicnrmse)}, std {np.std(total_Bicubicnrmse)}\n.")

#    print(f"total_Zerpadpsnr: ave {np.average(total_Zerpadpsnr)}, std {np.std(total_Zerpadpsnr)}\n"
#          f"total_Zerpadssim:  ave {np.average(total_Zerpadssim)}, std {np.std(total_Zerpadssim)}."
#          f"total_Zerpadnrmse:  ave {np.average(total_Zerpadnrmse)}, std {np.std(total_Zerpadnrmse)}\n.")





if __name__ == "__main__":
    main()
