# 这个文件的作用是进行格式转换，将数据集变为mnist的IDX格式存储
# 编写脚本将图像和标签转换为.ubyte 文件

import os
import numpy as np
from PIL import Image

def convert_to_idx(image_folder, output_prefix):
    images = []
    labels = []

    #读取所有的图像和标签
    for filename in os.listdir(image_folder):
        if filename.endswith('.png'):
            label = int(filename.split('_')[0])
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path).convert("L") #转为灰度
            img_array = np.array(img).reshape(28,28)
            images.append(img_array)
            labels.append(label)

    images_np = np.array(images, dtype=np.uint8)
    labels_np = np.array(labels, dtype=np.uint8) # 转换为numpy数组

    with open(f"{output_prefix}-images-idx3-ubyte", "wb") as f_img, \
         open(f"{output_prefix}-images-idx1-ubyte", "wb") as f_lbl:

        f_img.write(np.array[0,0,8,3], dtype=np.uint8.tobytes())

        f_img.write(np.array(images_np.shape[0], dtype=np.int32).tobytes())
        f_img.write(np.array(28, dtype=np.int32).tobytes())
        f_img.write(np.array(28, dtype=np.int32).tobytes())
        f_img.write(images_np.tobytes())

        f_lbl.write(np.array([0,0,8,1], dtype=np.uint8).tobytes())
        f_lbl.write(labels_np.tobytes())

convert_to_idx("custom_mnist", "custom_mnist")