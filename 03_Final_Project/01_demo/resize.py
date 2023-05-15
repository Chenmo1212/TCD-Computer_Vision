import cv2
import os

folder = '../02_Faster_Rrc/test_data'
save_path = '../02_Faster_Rrc/test_img'

# 判断保存路径是否存在，不存在就创建
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 读取文件夹下的所有图片
for filename in os.listdir(folder):
    img_path = os.path.join(folder, filename)
    img = cv2.imread(img_path)

    # 使用resize缩小一倍
    scale_percent = 0.5
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # 保存缩放后的图片
    save_name = os.path.join(save_path, filename)
    cv2.imwrite(save_name, resized)
    print('Save successfully:', save_name)
