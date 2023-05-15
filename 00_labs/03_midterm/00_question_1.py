import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def cv_show(name, img_data):
    cv2.imshow(name, img_data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_channels(img):
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]
    return R, G, B


def grayscale(img):
    h, w = img.shape[:2]  # 获取图片的high和wide
    gray = np.zeros([h, w], img.dtype)  # 创建一张和当前图片大小一样的单通道图片
    for i in range(h):
        for j in range(w):
            m = img[i, j]
            gray[i, j] = int(m[0] * 0.114 + m[1] * 0.587 + m[2] * 0.299)  # 将BGR坐标转换为gray坐标
    return gray


def binary(img, thresh):
    """
    :param img: img_gray
    :return: img
    """
    gray_img = grayscale(img)
    temp = np.where(gray_img >= thresh, 255, 0)
    return np.array(temp, dtype=np.uint8)


def brightness(img, num):
    h, w, chs = img.shape
    temp = np.zeros([h, w, chs], img.dtype)
    for i in range(h):
        for j in range(w):
            for c in range(chs):
                m = img[i, j, c]
                temp[i, j, c] = (m + num) if (m + num) <= 255 else 255
    return temp


def old_picture(img):
    """
    图像怀旧特效是指图像经历岁月的昏暗效果
    怀旧特效是将图像的RGB三个分量分别按照一定比例进行处理的结果，其怀旧公式如下所示：
    https://img-blog.csdnimg.cn/20201222224234588.png#pic_center
    :param img:
    :return:
    """
    h, w, chs = img.shape
    new_img = np.zeros((h, w, chs), dtype=img.dtype)

    for i in range(h):
        for j in range(w):
            temp_B = 0.272 * img[i, j][2] + 0.534 * img[i, j][1] + 0.131 * img[i, j][0]
            temp_G = 0.349 * img[i, j][2] + 0.686 * img[i, j][1] + 0.168 * img[i, j][0]
            temp_R = 0.393 * img[i, j][2] + 0.769 * img[i, j][1] + 0.189 * img[i, j][0]

            temp_B = temp_B if temp_B <= 255 else 255
            temp_G = temp_G if temp_G <= 255 else 255
            temp_R = temp_R if temp_R <= 255 else 255

            new_img[i, j] = np.uint8((temp_B, temp_G, temp_R))
    return new_img


def relief(img):
    """
    Python通过双层循环遍历图像的各像素点，使用相邻像素值之差来表示当前像素值，从而得到图像的边缘特征，最后加上固定数值150得到浮雕效果
    :param img:
    :return:
    """
    h, w, chs = img.shape
    new_img = np.zeros((h, w, chs), dtype=img.dtype)
    gray_img = grayscale(img)

    for i in range(h):
        for j in range(w - 1):
            gray0 = gray_img[i, j]
            gray1 = gray_img[i, j + 1]
            new_gray = int(gray0) - int(gray1) + 120

            if new_gray > 255:
                new_gray = 255
            elif new_gray < 0:
                new_gray = 0

            new_img[i, j] = new_gray

    return new_img


def contrast(img):
    h, w, chs = img.shape
    temp = np.zeros([h, w, chs], img.dtype)
    for i in range(h):
        for j in range(w):
            for c in range(chs):
                m = img[i, j, c]
                temp[i, j, c] = 10 * m ** 0.5
    return temp


def comics_effect(img):
    h, w, chs = img.shape
    new_img = np.zeros((h, w, chs), dtype=img.dtype)

    for i in range(w):
        for j in range(h):
            b = img[j, i, 0]
            g = img[j, i, 1]
            r = img[j, i, 2]
            R = int(int(abs(g - b + g + r)) * r / 256)
            G = int(int(abs(b - g + b + r)) * r / 256)
            B = int(int(abs(b - g + b + r)) * g / 256)
            new_img[j, i, 0] = R
            new_img[j, i, 1] = G
            new_img[j, i, 2] = B
    return new_img


def light(img, strength=200):
    """
    Python实现代码主要是通过双层循环遍历图像的各像素点，寻找图像的中心点，
    再通过计算当前点到光照中心的距离（平面坐标系中两点之间的距离），
    判断该距离与图像中心圆半径的大小关系，中心圆范围内的图像灰度值增强，
    范围外的图像灰度值保留，并结合边界范围判断生成最终的光照效果。
    :param img:
    :param strength:
    :return:
    """
    h, w, chs = img.shape
    new_img = np.zeros((h, w, chs), dtype=img.dtype)

    # 设置中心点
    center_x = h / 2
    center_y = w / 2
    radius = min(center_x, center_y)

    # 图像光照特效
    for i in range(h):
        for j in range(w):
            distance = math.pow((center_y - j), 2) + math.pow((center_x - i), 2)

            if (distance < radius * radius):
                result = int(strength * (1.0 - math.sqrt(distance) / radius))
                temp_B = img[i, j][0] + result
                temp_G = img[i, j][1] + result
                temp_R = img[i, j][2] + result

                temp_B = min(255, max(0, temp_B))
                temp_G = min(255, max(0, temp_G))
                temp_R = min(255, max(0, temp_R))
                new_img[i, j] = np.uint8((temp_B, temp_G, temp_R))
            else:
                temp_B = img[i, j][0]
                temp_G = img[i, j][1]
                temp_R = img[i, j][2]
                new_img[i, j] = np.uint8((temp_B, temp_G, temp_R))

    return new_img


def time(img):
    """
    流年是用来形容如水般流逝的光阴或年华，图像处理中特指将原图像转换为具有时代感或岁月沉淀的特效，其效果如下图所示。
    Python实现代码详见如下，它将原始图像的蓝色（B）通道的像素值开根号，再乘以一个权重参数，产生最终的流年效果。
    :param img:
    :return:
    """
    h, w, chs = img.shape
    new_img = np.zeros((h, w, chs), dtype=img.dtype)

    for i in range(h):
        for j in range(w):
            # B通道的数值开平方乘以参数12
            B = math.sqrt(img[i, j][0]) * 12
            G = img[i, j][1]
            R = img[i, j][2]
            if B > 255:
                B = 255
            new_img[i, j] = np.uint8((B, G, R))

    return new_img


def sketch(img_, gap=10):
    """
    素描滤镜的处理关键是对边缘的查找。通过对边缘的查找可以得到物体的线条感。
    在对图像进行灰度化处理后，我们首先需要确定一个阈值，这个需要根据自己去调整，这里我选用了10。
    我们知道素描主要强调的是明暗度的变化，绘制时是斜向方向，通过经验，我们将每个像素点的灰度值与其右下角的灰度值进行比较，
    当大于这个阈值时，就判断其是轮廓并绘制。
    :param img_:
    :return:
    """
    gray_img = grayscale(img_)
    h, w = gray_img.shape
    new_img = np.zeros((h, w), dtype=gray_img.dtype)

    for i in range(h - 1):
        for j in range(w - 1):
            curr = gray_img[i, j]
            next = gray_img[i + 1, j + 1]

            diff = abs(int(curr) - int(next))
            new_img[i, j] = 0 if diff >= gap else 255

    return new_img


def mosaic(img):
    """
    马赛克特效，是当前使用较为广泛的一种图像或视频处理手段，它将图像或视频中特定区域的色阶细节劣化并造成色块打乱的效果，
    主要目的通常是使特定区域无法辨认。其数学原理很简单，就是让某个集合内的像素相同即可
    :param img:
    :return:
    """
    h, w, chs = img.shape
    new_img = np.zeros((h, w, chs), dtype=img.dtype)

    for i in range(h - 5):
        for j in range(w - 5):
            if i % 5 == 0 and j % 5 == 0:
                for k in range(5):
                    for t in range(5):
                        new_img[i + k, j + t] = img[i, j]

    return new_img


def draw_subplot(name, img_, index):
    plt.subplot(4, 2, index)
    plt.title(name, fontsize=16)
    plt.axis('off')

    if len(img_.shape) == 3:
        plt.imshow(img_[:, :, ::-1])
    else:
        plt.imshow(img_, cmap='gray')


if __name__ == '__main__':
    base_url = './images/Q1/'
    # 1.0
    # img = cv2.imread('./lights.png')
    # img = cv2.imread('./men.png')
    # img = cv2.imread('./apartment.png')
    # img = cv2.imread('./cat.png')

    # img = cv2.imread('./1.1.png')
    # img = cv2.imread('./1.2.png')
    # img = cv2.imread('./1.3.png')
    img = cv2.imread(base_url + '1.5.png')

    # img_gray = grayscale(img)
    # cv_show('gray-img', img_gray)

    # 1.1
    img_binary = binary(img, 115)
    # cv_show('binary-img', img_binary)

    # img_bri = brightness(img, 50)
    # cv_show('brightness-img', img_bri)

    # 1.2
    img_old = old_picture(img)
    # cv_show('old-img', img_old)

    # 1.3
    # img_relief = relief(img)
    # cv_show('relief-img', img_relief)

    # img_cont = contrast(img)
    # cv_show('contrast-img', img_cont)

    # img_com = comics_effect(img)
    # cv_show('comics_effect-img', img_com)

    # 1.4
    # img_light = light(img, 80)
    # cv_show('light-img', img_light)

    # 1.5
    # img_time = time(img)
    # cv_show('time-img', img_time)

    # 1.6
    img_sketch = sketch(img, 5)
    cv_show('sketch-img', img_sketch)

    # 1.7
    # img_mosaic = mosaic(img)
    # cv_show('mosaic-img', img_mosaic)

    # plt.rcParams['figure.figsize'] = (10, 18)
    # figure, axes = plt.subplots(4, 2)
    # figure.tight_layout()  # 调整整体空白
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)  # 调整子图间距
    #
    # draw_subplot('Origin', img, 1)
    # draw_subplot('Binary', img_binary, 2)
    # draw_subplot('Relief', img_relief, 3)
    # draw_subplot('Sketch', img_sketch, 4)
    # draw_subplot('Old', img_old, 5)
    # draw_subplot('Time', img_time, 6)
    # draw_subplot('Light', img_light, 7)
    # draw_subplot('Mosaic', img_mosaic, 8)
    #
    # plt.savefig('1.5-result.png', bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    #
    # print('滤镜生成完成')
