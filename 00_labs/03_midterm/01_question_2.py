import time

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def cv_show(name, img_data):
    cv2.imshow(name, img_data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cv_save(name, img_data):
    cv2.imwrite(name, img_data)


def filter(img_, k):
    h, w, chs = img_.shape
    size = len(k)

    pad = size // 2
    new_img = np.zeros((h + 2 * pad, w + 2 * pad, chs), dtype=img_.dtype)
    new_img[pad:pad + h, pad:pad + w] = img_.copy().astype(img_.dtype)

    temp = new_img.copy()
    for y in range(h):
        for x in range(w):
            for c in range(chs):
                res = np.sum(k * temp[y:y + size, x:x + size, c])
                new_img[pad + y, pad + x, c] = np.clip(res, 0, 255)

    new_img = new_img[pad:pad + h, pad:pad + w].astype(img_.dtype)

    return new_img


def median(img_, size=5):
    h, w, chs = img_.shape

    pad = size // 2
    new_img = np.zeros((h + 2 * pad, w + 2 * pad, chs), dtype=img_.dtype)
    new_img[pad:pad + h, pad:pad + w] = img_.copy().astype(img_.dtype)

    temp = new_img.copy()
    for y in range(h):
        for x in range(w):
            for c in range(chs):
                new_img[pad + y, pad + x, c] = np.median(temp[y:y + size, x:x + size, c])

    new_img = new_img[pad:pad + h, pad:pad + w].astype(img_.dtype)
    return new_img


# 双边滤波
# radius:滤波器窗口半径
# sigma_color:颜色域方差
# sigma_space:空间域方差
def bilateral(img_, r=5, sc=15, sp=10):
    """
    结合了图像的空间邻近度和像素值相似度（即空间域和值域）的一种折中处理，从而达到保边去噪的目的。
    :param img_:
    :param r: filter radius
    :param sc: sigma_color: color gamut variance
    :param sp: sigma_space: Spatial domain variance
    :return:
    """
    h, w, chs = img_.shape
    new_img = np.zeros((h, w, chs), dtype=img_.dtype)

    for i in range(r, h - r):
        for j in range(r, w - r):
            for k in range(chs):

                weight_sum = 0.0
                pixel_sum = 0.0
                for x in range(-r, r + 1):
                    for y in range(-r, r + 1):
                        # Spatial Domain Weights
                        spatial_w = -(x ** 2 + y ** 2) / (2 * (sp ** 2))
                        # color gamut weights
                        color_w = -(int(img_[i][j][k]) - int(img_[i + x][j + y][k])) ** 2 / (2 * (sc ** 2))
                        # Pixel overall weight
                        weight = np.exp(spatial_w + color_w)
                        # Sum of weights for normalization
                        weight_sum += weight
                        pixel_sum += (weight * img_[i + x][j + y][k])
                value = pixel_sum / weight_sum
                new_img[i][j][k] = value
    return new_img.astype(np.uint8)


# smoothing
# ============================================
# Mean Blur
kernel_mean = (1 / 9.0) * np.array([[1, 1, 1],
                                    [1, 1, 1],
                                    [1, 1, 1]])

# Gaussian Blur 5x5
kernel_gaussian = (1 / 256.0) * np.array([[1, 4, 6, 4, 1],
                                          [4, 16, 24, 16, 4],
                                          [6, 24, 36, 24, 6],
                                          [4, 16, 24, 16, 4],
                                          [1, 4, 6, 4, 1]])

# bilateral

# sharping
# ============================================
# Unsharp masking 5x5
kernel_unsharp = -(1 / 256.0) * np.array([[1, 4, 6, 4, 1],
                                          [4, 16, 24, 16, 4],
                                          [6, 24, -476, 24, 6],
                                          [4, 16, 24, 16, 4],
                                          [1, 4, 6, 4, 1]])

# Sharpen
kernel_sharpen = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

# Emboss
kernel_emboss = np.array([[-2, -1, 0],
                          [-1, 1, 1],
                          [0, 1, 2]])

if __name__ == '__main__':
    base_url = './images/Q2/'
    file_name = '2.8'
    img = cv2.imread(base_url + file_name + '.png')

    kernels = [kernel_sharpen, kernel_emboss, kernel_unsharp, kernel_mean, kernel_gaussian]
    kernel_name = ['Sharpen', 'Emboss', '5x5 Unsharp Masking', 'Mean', '5x5 Gaussian Blur']
    kernel_res = []

    start = time.time()

    figure, axes = plt.subplots(4, 2, figsize=(7, 14))
    figure.tight_layout()  # 调整整体空白
    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # 调整子图间距

    # add Origin
    plt.subplot(4, 2, 1)
    plt.imshow(img[:, :, ::-1])
    plt.axis('off')
    plt.title('Origin')

    # add kernels
    index = 2
    for kernel, name in zip(kernels, kernel_name):
        print('start {}'.format(name))
        conv_im = filter(img, kernel[::-1, ::-1])
        plt.subplot(4, 2, index)
        plt.imshow(abs(conv_im[:, :, ::-1]))
        plt.title(name)
        plt.axis('off')
        index += 1

    # add bilateral
    print('start bilateral')
    img_bilateral = bilateral(img)
    plt.subplot(4, 2, 7)
    plt.imshow(img_bilateral[:, :, ::-1])
    plt.title('Bilateral')
    plt.axis('off')

    # add median
    print('start median')
    img_median = median(img)
    plt.subplot(4, 2, 8)
    plt.imshow(img_median[:, :, ::-1])
    plt.title('Median')
    plt.axis('off')

    plt.savefig(base_url + '{}-result.png'.format(file_name), bbox_inches='tight', pad_inches=0.1)
    print('done')

    end = time.time()
    print('执行时间 = {} min {} s'.format(int((end - start) / 60), int((end - start) % 60)))

    plt.show()