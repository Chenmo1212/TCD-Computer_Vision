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


def get_channels(img):
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]
    return R, G, B


def add_noise(img_, num=1000):
    h, w, chs = img_.shape

    # add noise
    for i in range(1000):
        x = np.random.randint(0, h)
        y = np.random.randint(0, w)
        img_[x, y, :] = 255

    return img_


def nearest(img_, scale):
    h, w, chs = img_.shape
    new_img = np.zeros((int(h * scale), int(w * scale), chs), dtype=img_.dtype)

    for i in range(int(scale * h)):
        for j in range(int(scale * w)):
            new_img[i, j] = img_[int(i / scale), int(j / scale)]
    return new_img


def bilinear(img_, scale):
    # scrH,scrW,_ = img_.shape
    h, w, chs = img_.shape

    # Extend the height and width of the original image by one pixel
    # The purpose is to prevent the array out of bounds in subsequent calculations.
    img_ = np.pad(img_, ((0, 1), (0, 1), (0, 0)), 'constant')
    new_img = np.zeros((int(scale * h), int(scale * w), chs), dtype=np.uint8)
    for i in range(int(h * scale)):
        for j in range(int(w * scale)):
            for c in range(chs):
                scrx = (i + 1) * (1 / scale) - 1
                scry = (j + 1) * (1 / scale) - 1
                x = int(scrx)
                y = int(scry)
                u = scrx - x
                v = scry - y
                new_img[i, j] = (1 - u) * (1 - v) * img_[x, y] + \
                                u * (1 - v) * img_[x + 1, y] + \
                                (1 - u) * v * img_[x, y + 1] + \
                                u * v * img_[x + 1, y + 1]
    return new_img


def W(x):
    """
    Generate 16 pixels with different weights
    """
    x = abs(x)
    if x <= 1:
        return 1 - 2 * (x ** 2) + (x ** 3)
    elif x < 2:
        return 4 - 8 * x + 5 * (x ** 2) - (x ** 3)
    else:
        return 0


def bi_cubic(img_, scale):
    h, w, chs = img_.shape
    img_ = np.pad(img_, ((1, 3), (1, 3), (0, 0)), 'constant')
    new_img = np.zeros((int(scale * h), int(scale * w), chs), dtype=np.uint8)

    for i in range(int(scale * h)):
        for j in range(int(scale * w)):
            scrx = i * (1 / scale)
            scry = j * (1 / scale)
            x = math.floor(scrx)
            y = math.floor(scry)
            u = scrx - x
            v = scry - y
            tmp = 0

            for ii in range(-1, 2):
                for jj in range(-1, 2):
                    if x + ii < 0 or y + jj < 0 or x + ii >= h or y + jj >= w:
                        continue
                    tmp += img_[x + ii, y + jj] * W(ii - u) * W(jj - v)
            new_img[i, j] = np.clip(tmp, 0, 255)

    return new_img


def gaussian_filter(img_, k, size=5):
    h, w, chs = img_.shape

    pad = size // 2
    new_img = np.zeros((h + 2 * pad, w + 2 * pad, chs), dtype=img_.dtype)
    new_img[pad:pad + h, pad:pad + w] = img_.copy().astype(img_.dtype)

    temp = new_img.copy()
    for y in range(h):
        for x in range(w):
            for c in range(chs):
                new_img[pad + y, pad + x, c] = np.sum(k * temp[y:y + size, x:x + size, c])

    new_img = new_img[pad:pad + h, pad:pad + w].astype(img_.dtype)

    return new_img


def insertZero(img_):
    h, w, chs = img_.shape
    new_img = np.zeros((2 * h, 2 * w, chs), dtype=img_.dtype)
    new_img[::2, ::2] = img_
    return new_img


def gaussian(img_, scale, k):
    """
    minify or magnify 2x of image
    :return:
    """
    if scale >= 1:
        new_img = insertZero(img_)
        new_img = gaussian_filter(new_img, k)
    else:
        img_gaussian = gaussian_filter(img_, k)
        # Only keep "one pixel in 1 / scale" in both dimensions
        new_img = img_gaussian[::2, ::2]
    return new_img


def LanczosInter(img_, scale, a=2):
    h, w, chs = img_.shape
    new_img = np.zeros((scale * h, scale * w, chs), dtype=img_.dtype)
    img_ = np.pad(img_, ((1, 1), (1, 1), (0, 0)), 'constant')

    for i in range(int(h * scale)):
        for j in range(int(w * scale)):
            for k in range(chs):
                i_ = int(i / scale) - a + 1
                j_ = int(j / scale) - a + 1
                temp = 0

                for m in range(i_, i_ + 2 * a):
                    for n in range(j_, j_ + 2 * a):
                        temp += LW(a, i / scale - m) * LW(a, j / scale - n) * img_[m + a][n + a][k]

                new_img[i][j][k] = np.clip(temp, 0, 255)
    return new_img


def LW(a, x):
    w = 0
    if x == 0:
        w = 1
    elif abs(x) < a:
        w = a * math.sin(math.pi * x) * math.sin(math.pi * x / a) / pow(math.pi * x, 2)
    return w


def draw_subplot(name, img_, index):
    plt.subplot(2, 2, index)
    plt.title(name, fontsize=16)
    plt.axis('off')

    if len(img_.shape) == 3:
        plt.imshow(img_[:, :, ::-1])
    else:
        plt.imshow(img_, cmap='gray')


kernel_gaussian = (1 / 64.0) * np.array([[1, 4, 6, 4, 1],
                                         [4, 16, 24, 16, 4],
                                         [6, 24, 36, 24, 6],
                                         [4, 16, 24, 16, 4],
                                         [1, 4, 6, 4, 1]])

if __name__ == '__main__':
    base_url = './images/Q3/'

    img_name = '3.1'  # 7, 10, minify
    # img_name = '3.2'  # 10, 8, minify
    img_name = '3.3'  # 10, 8, magnify
    # img_name = '3.4'  # 7, 10, magnify
    img = cv2.imread(base_url + img_name + '.png')
    # cv_save(base_url + '{}-origin.png'.format(img_name), img)

    scale = 8

    print('start nearest')
    # img_near = nearest(img, scale)
    # cv_show('img_near', img_near)
    # cv_save(base_url + '{}-near-{}.png'.format(img_name, scale), img_near)

    print('start bilinear')
    # img_bilinear = bilinear(img, scale)
    # cv_show('img_bi_linear', img_bi_linear)
    # cv_save(base_url + '{}-bilinear-{}.png'.format(img_name, scale), img_bilinear)

    print('start bicubic')
    img_bicubic = bi_cubic(img, scale)
    # cv_show('img_bicubic', img_bicubic)
    # cv_save(base_url + '{}-bicubic-{}.png'.format(img_name, scale), img_bicubic)

    # ================================================
    # magnify
    #

    print('start lanczos')
    start = time.time()
    img_lanczos = LanczosInter(img, scale, 2)
    end = time.time()
    print('执行时间 = {} min {} s'.format(int((end - start) / 60), int((end - start) % 60)))
    # cv_show('img_lanczos', img_lanczos)
    # cv_save(base_url + '{}-lanczos-{}.png'.format(img_name, scale), img_lanczos)

    print('start gaussian')
    img_gaussian = gaussian(img, 2, kernel_gaussian)
    img_gaussian = gaussian(img_gaussian, 4, kernel_gaussian)
    img_gaussian = gaussian(img_gaussian, 8, kernel_gaussian)
    # cv_show('img_gaussian', img_gaussian)
    # cv_save(base_url + '{}-Gaussian-{}.png'.format(img_name, scale), img_gaussian)

    plt.rcParams['figure.figsize'] = (10, 8)
    # plt.rcParams['figure.figsize'] = (7, 10)
    figure, axes = plt.subplots(2, 2)
    figure.tight_layout()  # 调整整体空白
    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # 调整子图间距

    a = ['Nearest_neighbor', 'Bilinear', 'Bicubic', 'Bilateral', 'Lanczos']

    draw_subplot('Origin', img, 1)
    # draw_subplot('Nearest_neighbor-{}'.format(scale), img_near, 2)
    # draw_subplot('Bilinear-{}'.format(scale), img_bilinear, 3)
    # draw_subplot('Bicubic-{}'.format(scale), img_bicubic, 4)
    draw_subplot('Bicubic-{}'.format(scale), img_bicubic, 2)
    draw_subplot('Lanczos-{}'.format(scale), img_lanczos, 3)
    draw_subplot('Gaussian-{}'.format(scale), img_gaussian, 4)
    plt.savefig(base_url + '{}-{}-result.png'.format(img_name, scale), bbox_inches='tight', pad_inches=0.1)
    plt.show()

    print('done')
