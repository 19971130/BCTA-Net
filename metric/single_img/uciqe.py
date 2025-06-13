import os

import cv2
import numpy as np
from PIL import Image


def calc_uciqe(img: Image, nargin: int = 1):
    img_clone = img.copy()
    img_bgr = cv2.cvtColor(np.array(img_clone), cv2.COLOR_RGB2BGR)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)  # Transform to Lab color space

    if nargin == 1:  # According to training result mentioned in the paper:
        coe_metric = [0.4680, 0.2745, 0.2576]  # Obtained coefficients are: c1=0.4680, c2=0.2745, c3=0.2576.
    img_lum = img_lab[..., 0] / 255
    img_a = img_lab[..., 1] / 255
    img_b = img_lab[..., 2] / 255

    img_chr = np.sqrt(np.square(img_a) + np.square(img_b))  # Chroma

    img_sat = img_chr / np.sqrt(np.square(img_chr) + np.square(img_lum))  # Saturation
    aver_sat = np.mean(img_sat)  # Average of saturation

    aver_chr = np.mean(img_chr)  # Average of Chroma

    var_chr = np.sqrt(np.mean(abs(1 - np.square(aver_chr / img_chr))))  # Variance of Chroma

    dtype = img_lum.dtype  # Determine the type of img_lum
    if dtype == 'uint8':
        nbins = 256
    else:
        nbins = 65536

    hist, bins = np.histogram(img_lum, nbins)  # Contrast of luminance
    cdf = np.cumsum(hist) / np.sum(hist)

    ilow = np.where(cdf > 0.0100)
    ihigh = np.where(cdf >= 0.9900)
    tol = [(ilow[0][0] - 1) / (nbins - 1), (ihigh[0][0] - 1) / (nbins - 1)]
    con_lum = tol[1] - tol[0]

    quality_val = coe_metric[0] * var_chr + coe_metric[1] * con_lum + coe_metric[
        2] * aver_sat  # get final quality value
    return np.float(quality_val)


if __name__ == '__main__':
    root = r'F:\temp\a\p_results\p_results\01_UIEB\02test\mp'
    total = 0
    count = 0
    for filename in os.listdir(root):
        img = Image.open(os.path.join(root, filename))
        uciqe = calc_uciqe(img)
        total += uciqe
        count += 1
        print("{}: {:.6f}".format(filename, calc_uciqe(img)))
    print('ave: {}'.format(total / count))