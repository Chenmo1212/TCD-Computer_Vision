import cv2

# def imread(filePath, *config)
# config: cv2.IMREAD_UNCHANGED ：表示和原图像一致
#         cv2.IMREAD_GRAYSCALE : 表示将原图像转化为灰色图像。
#         cv2.IMREAD_COLOR：表示将原图像转化为彩色图像。
img = cv2.imread('./opencv.png', cv2.IMREAD_UNCHANGED)  # path指图片相关路径

# def imshow(windowName, imgName)
# cv2.imshow('hahaha', img)

# 如果没有这个限制，那么显示的图像就会一闪而过
# dealy=0,无限等待图像显示，直到关闭。也是waitKey的默认数值。
# delay<0,等待键盘点击结束图像显示，也就是说当我们敲击键盘的时候，图像结束显示。
# delay>0,等待delay毫秒后结束图像显示。
# cv2.waitKey(0)

# 把图像从内存中彻底删除
# cv2.destroyAllWindows()

# 保存图片
# cv2.imwrite('test.jpg', img)

# 读取像素1: 直接根据坐标
# p1 = img[10, 10]
# print(p1)
# # 读取像素2: 利用函数.item
# p2 = img.item(10, 10)
# print(p2)

# 设置像素， 图像名.itemset（位置，新的数值）
# img.itemset((10, 10), 0)

print(img.shape)

# 拆分通道
# b = img[:, :, 0]
# g = img[:, :, 1]
# r = img[:, :, 2]

# 拆分通道的函数
b, g, r = cv2.split(img)
# cv2.imshow("B", b)
# cv2.imshow("G", g)
# cv2.imshow("R", r)

# 合并通道
m = cv2.merge([b, g, r])

# 类型转换
# b = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', b)

# 图像缩放: dst=cv2.resize(src,dsize)
# re_img = cv2.resize(img, (200, 200))
# re_img = cv2.resize(img, None, fx=1.2, fy=0.5)
# cv2.imshow('resize', re_img)

# 图像翻转dst=cv2.flip(src,flipcode)
# flipcode=0；表示以x轴为对称轴上下翻转。
# flipcode>0；表示以y轴为对称轴上下翻转。
# flipcode<0；表示以x轴和y轴为对称轴同时翻转。
# dst = cv2.flip(img, 1)
# cv2.imshow('翻转', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
