import os
import math
import numpy as np
import cv2
import glob
from natsort import natsorted
from skimage.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def main():

    # GT - Ground-truth;
    # Gen: Generated / Restored / Recovered images
    folder_GT = 'C:/HyyData/hinet/image/secret'
    folder_Gen = 'C:/HyyData/hinet/image/secret-rev'

    crop_border = 1
    suffix = '_secret_rev'  # suffix for Gen images
    test_Y = True  # True: test Y channel only; False: test RGB channels

    PSNR_all = []
    SSIM_all = []
    # Configurations
    RMSE_all = []
    MAE_all = []
    img_list = sorted(glob.glob(folder_GT + '/*'))
    img_list = natsorted(img_list)

    if test_Y:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')

    for i, img_path in enumerate(img_list):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        # base_name = base_name[:5]
        im_GT = cv2.imread(img_path) / 255.
        # print(base_name)
        # print(img_path)
        # print(os.path.join(folder_Gen, base_name + '.png'))
        im_Gen = cv2.imread(os.path.join(folder_Gen, base_name + '.png')) / 255.


        if test_Y and im_GT.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
            im_GT_in = bgr2ycbcr(im_GT)
            im_Gen_in = bgr2ycbcr(im_Gen)

        else:
            im_GT_in = im_GT
            im_Gen_in = im_Gen

        # # crop borders
        # if im_GT_in.ndim == 3:
        #     cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border, :]
        #     cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border, :]
        # elif im_GT_in.ndim == 2:
        #     cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border]
        #     cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border]
        # else:
        #     raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im_GT_in.ndim))

        # calculate PSNR and SSIM
        RMSE = calculate_rmse(im_GT_in * 255, im_Gen_in * 255)
        PSNR = calculate_psnr(im_GT_in * 255, im_Gen_in * 255)
        SSIM = calculate_ssim(im_GT_in * 255, im_Gen_in * 255)
        MAE = calculate_mae(im_GT_in * 255, im_Gen_in * 255)
        print('{:3d} - {:25}. \tPSNR: {:.6f} dB, \tSSIM: {:.6f}, \tRMSE: {:.6f}, \tMAE:{:.6f}'.format(
            i + 1, base_name, PSNR, SSIM, RMSE, MAE))
        PSNR_all.append(PSNR)
        SSIM_all.append(SSIM)
        RMSE_all.append(RMSE)
        MAE_all.append(MAE)
    print('Average: PSNR: {:.6f} dB, SSIM: {:.6f}, RMSE:{:.6f}, MAE:{:.6f}'.format(
        sum(PSNR_all) / len(PSNR_all),
        sum(SSIM_all) / len(SSIM_all),
        sum(RMSE_all) / len(RMSE_all),
        sum(MAE_all) / len(MAE_all)))

    with open('1.txt', 'w') as f:
        f.write(str(PSNR_all))

def calculate_rmse(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    rmse = mean_squared_error(img1,img2)
    return rmse

def calculate_mae(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mae = mean_absolute_error(img1,img2)
    return mae

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def bgr2ycbcr(img, only_y=True):

    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


if __name__ == '__main__':
    main()
