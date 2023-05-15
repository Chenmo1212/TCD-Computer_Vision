import os
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
import cv2
from glob import glob


def get_img_size(img):
    img = cv2.imread(img)
    return img.shape[0], img.shape[1]


def txt_to_xml(txt_path, xml_path):
    # Read txt file
    with open(txt_path, 'r') as txt_file:
        lines = txt_file.readlines()
    # Extracting Information
    ripeness_class = lines[0].split()[0]
    x_center = lines[0].split()[1]
    y_center = lines[0].split()[2]
    x_width = lines[0].split()[3]
    y_width = lines[0].split()[4]
    ripeness_text = 'unripe' if ripeness_class == '0' else 'partially ripe' if ripeness_class == '1' else 'fully ripe'

    # Create XML elements
    annotation = Element('annotation')
    SubElement(annotation, 'filename').text = os.path.basename(txt_path).replace('.txt', '.png')
    size = SubElement(annotation, 'size')
    SubElement(size, 'width').text = str(W)
    SubElement(size, 'height').text = str(H)
    SubElement(size, 'depth').text = '3'
    object_ = SubElement(annotation, 'object')
    SubElement(object_, 'name').text = ripeness_text
    bndbox = SubElement(object_, 'bndbox')
    SubElement(bndbox, 'xmin').text = str(int(float(x_center) * W - float(x_width) * W / 2))
    SubElement(bndbox, 'ymin').text = str(int(float(y_center) * H - float(y_width) * H / 2))
    SubElement(bndbox, 'xmax').text = str(int(float(x_center) * W + float(x_width) * W / 2))
    SubElement(bndbox, 'ymax').text = str(int(float(y_center) * H + float(y_width) * H / 2))
    # 写入 XML 文件
    with open(xml_path, 'w') as xml_file:
        xml_file.write(tostring(annotation).decode())


H, W = get_img_size('train_data/1.png')

folder_path = 'train_data'
file_names = [os.path.basename(x) for x in glob(folder_path + '/*')]

# Convert all txt files
txt_dir = '01_bounding_box/'
xml_dir = 'train_xml/'
for txt_filename in file_names:
    txt_path = os.path.join(txt_dir, txt_filename.replace('.png', '.txt'))
    xml_path = os.path.join(xml_dir, txt_filename.replace('.png', '.xml'))
    # print(txt_path, xml_path)
    txt_to_xml(txt_path, xml_path)
