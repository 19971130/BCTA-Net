import os
import cv2
from PIL import Image
import pandas as pd
from metric.single_img.uiqm import calc_uiqm
from metric.single_img.uciqe import calc_uciqe
from metric.single_img.niqe import calc_niqe
from metric.double_img.psnr import calc_psnr
from metric.double_img.mse import calc_mse
from metric.double_img.ssim import calc_ssim


def calc_one_single_img_metrics(img_path: str, precision: int = 3):
    res = {'NAME': img_path.split('\\')[-1].split('.')[0]}
    img = Image.open(img_path)
    res['UIQM'] = round(calc_uiqm(img), precision)
    print('uiqm finish')
    res['UCIQE'] = round(calc_uciqe(img), precision)
    print('uciqe finish')
    res['NIQE'] = round(calc_niqe(img), precision)
    print('niqe finish')
    return res


def calc_multi_single_img_metrics(img_dir: str):
    single_metrics = {'NAME': [], 'UIQM': [], 'UCIQE': [], 'NIQE': []}
    filenames = os.listdir(img_dir)
    filenames.sort(key=lambda x: int(x.split('.')[0]))
    for filename in filenames:
        img_path = os.path.join(img_dir, filename)
        print('img_path\t' + img_path)
        one_img_metric = calc_one_single_img_metrics(img_path)
        for key in single_metrics.keys():
            single_metrics[key].append(one_img_metric[key])
    return single_metrics


def calc_one_double_img_metrics(prd_img_path: str, tar_img_path, precision: int = 3):
    res = {'NAME': prd_img_path.split('\\')[-1].split('.')[0]}
    prd = Image.open(prd_img_path)
    tar = Image.open(tar_img_path)
    if prd.size != tar.size:
        prd = prd.resize(tar.size, Image.BILINEAR)
    res['PSNR'] = round(calc_psnr(tar, prd), precision)
    print('psnr finish')
    res['MSE'] = round(calc_mse(tar, prd), precision)
    print('mse finish')
    res['SSIM'] = round(calc_ssim(tar, prd), precision)
    print('ssim finish')
    return res


def calc_multi_double_img_metrics(prd_img_dir: str, tar_img_dir: str):
    double_metrics = {'NAME': [], 'PSNR': [], 'MSE': [], 'SSIM': []}
    filenames = os.listdir(prd_img_dir)
    filenames.sort(key=lambda x: int(x.split('.')[0]))
    for filename in filenames:
        prd_img_path = os.path.join(prd_img_dir, filename)
        tar_img_path = os.path.join(tar_img_dir, filename)
        print('prd_path:\t' + prd_img_path)
        print('tar_path:\t' + tar_img_path)
        one_img_metric = calc_one_double_img_metrics(prd_img_path, tar_img_path)
        for key in double_metrics.keys():
            double_metrics[key].append(one_img_metric[key])
    return double_metrics


def record_metric(metrics: dict, xlsx_path: str, sheet_name: str) -> None:
    df = pd.DataFrame(metrics)
    if not os.path.exists(xlsx_path):
        df.to_excel(xlsx_path, sheet_name=sheet_name, index=False)
    else:
        writer = pd.ExcelWriter(xlsx_path, mode='a', engine='openpyxl')
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        writer.save()
        writer.close()


def calc_one_datasets(prd_root: str, tar_root: str, test_256: bool, xlsx_name:str, save_path: str = './'):
    if test_256:
        xlsx_name += '_256'
    img_size = '256'
    # dataset = {'UIEB VAL': r'01_UIEB\01_val', 'UIEB TEST': r'01_UIEB\02_test', 'UCCS BLUE': r'02_UCCS_300\01_Blue',
    #            'UCCS BLUE GREEN': r'02_UCCS_300\02_Blue_Green', 'UCCS GREEN': r'02_UCCS_300\03_Green',
    #            'MABLs': r'03_MABLs', 'UFO120': r'04_UFO120', 'EUVP': r'05_EUVP', 'U45': r'06_U45'}
    # dataset = { 'UIEB VAL': r'01_UIEB\01_val'}
    dataset = '01_UIEB/01_val'
    double_metrics = {}
    tmp_path = os.path.join(prd_root, '256', dataset)
    single_metrics = calc_multi_single_img_metrics(tmp_path)

    tar_path = os.path.join(tar_root, img_size, dataset)
    double_metrics = calc_multi_double_img_metrics(tmp_path, tar_path)
    if len(double_metrics) == 0:
        res = single_metrics
    else:
        res = {**single_metrics, **double_metrics}
    # xlsx_name = prd_root.split('\\')[-1].split('_')[1]
    # xlsx_name = xlsx_name
    record_metric(res, os.path.join(save_path, xlsx_name + '.xlsx'), 'UIEB VAL')


if __name__ == '__main__':
    # 计算一个模型所有数据集指标
    tar_root = r'C:\Users\Lenovo\Desktop\Tar'
    prd_root = r'C:\Users\Lenovo\Desktop\Ours'

    calc_one_datasets(prd_root, tar_root, test_256=True, xlsx_name='00')
