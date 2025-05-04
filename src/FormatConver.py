import os
import numpy as np
from PIL import Image

#格式校验
def convert_to_idx(image_folder, output_prefix):
    images = []
    labels = []
    for filename in os.listdir(image_folder):
        if filename.endswith(".png"):
            label = int(filename.split("_")[0])
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path).convert("L")

            # 强制调整图像为 28x28
            if img.size != (28, 28):
                img = img.resize((28, 28))

            img_array = np.array(img).reshape(28, 28)
            images.append(img_array)
            labels.append(label)


def convert_to_idx(image_folder, output_prefix):
    images = []
    labels = []

    # 读取所有的图像和标签
    for filename in os.listdir(image_folder):
        if filename.endswith('.png'):
            label = int(filename.split('_')[0])
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path).convert("L")  # 转为灰度
            img_array = np.array(img).reshape(28, 28)
            images.append(img_array)
            labels.append(label)

    images_np = np.array(images, dtype=np.uint8)
    labels_np = np.array(labels, dtype=np.uint8)

    with open(f"{output_prefix}-images-idx3-ubyte", "wb") as f_img, \
         open(f"{output_prefix}-labels-idx1-ubyte", "wb") as f_lbl:

        # 写入图像文件头
        f_img.write(np.array([0, 0, 8, 3], dtype=np.uint8).tobytes())  # Magic number
        f_img.write(np.array(images_np.shape[0], dtype='>i4').tobytes())  # 大端32位整数
        f_img.write(np.array(28, dtype='>i4').tobytes())
        f_img.write(np.array(28, dtype='>i4').tobytes())
        f_img.write(images_np.tobytes())

        # 写入标签文件头
        f_lbl.write(np.array([0, 0, 8, 1], dtype=np.uint8).tobytes())  # Magic number
        f_lbl.write(np.array(labels_np.shape[0], dtype='>i4').tobytes())
        f_lbl.write(labels_np.tobytes())

convert_to_idx("custom_mnist2", "custom_mnist2")