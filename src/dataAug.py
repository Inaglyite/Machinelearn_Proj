import os
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
import random


# 定义增强函数（每个函数返回增强后的图像和操作后缀名）
def rotate_image(image, degrees=10):
    """旋转增强（保持28x28尺寸）"""
    img = image.rotate(degrees, fillcolor=255)
    img = img.resize((28, 28))  # 强制调整尺寸
    suffix = f"_rotate_{degrees}"
    return img, suffix

def shift_image(image, dx=2, dy=2):
    """平移增强（保持28x28尺寸）"""
    img = ImageOps.expand(image, border=(dx, dy, 0, 0), fill=255)
    img = img.resize((28, 28))  # 强制调整尺寸
    suffix = f"_shift_{dx}_{dy}"
    return img, suffix

def add_noise(image, noise_level=20):
    """添加噪声（保持28x28尺寸）"""
    img_array = np.array(image).astype(np.float32)
    noise = np.random.normal(0, noise_level, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array).resize((28, 28))  # 强制调整尺寸
    suffix = f"_noise_{noise_level}"
    return img, suffix


def random_augment(image):
    """随机组合增强（自动生成复合后缀）"""
    ops = []
    img = image

    # 随机旋转 -15°到15°
    if random.random() > 0.5:
        degrees = random.randint(-15, 15)
        img, suffix = rotate_image(img, degrees)
        ops.append(suffix)

    # 随机平移 -5到5像素
    if random.random() > 0.5:
        dx = random.randint(-5, 5)
        dy = random.randint(-5, 5)
        img, suffix = shift_image(img, dx, dy)
        ops.append(suffix)

    # 随机噪声
    if random.random() > 0.5:
        noise_level = random.randint(10, 30)
        img, suffix = add_noise(img, noise_level)
        ops.append(suffix)

    return img, "_".join(ops)


# 主处理函数
def augment_dataset(input_dir="custom_mnist2", output_dir="custom_mnist2"):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".png"):
            continue

        # 解析原始文件名（如 "5_001.png"）
        try:
            label, serial = filename.split("_")[:2]
            serial = serial.split(".")[0]
            if not label.isdigit() or int(label) not in range(10):
                continue
        except:
            print(f"跳过无效文件名: {filename}")
            continue

        # 加载原始图像
        img_path = os.path.join(input_dir, filename)
        image = Image.open(img_path).convert("L")  # 转为灰度

        # 应用所有增强操作（可自定义需要哪些增强）
        augmented_images = [
            rotate_image(image, degrees=10),
            rotate_image(image, degrees=-10),
            shift_image(image, dx=2, dy=2),
            shift_image(image, dx=-2, dy=-2),
            add_noise(image, noise_level=20),
            random_augment(image)  # 随机组合增强
        ]

        # 保存所有增强后的图像
        for aug_img, suffix in augmented_images:
            new_filename = f"{label}_{serial}{suffix}.png"
            new_path = os.path.join(output_dir, new_filename)
            aug_img.save(new_path)
            print(f"生成增强图像: {new_path}")


if __name__ == "__main__":
    augment_dataset(input_dir="custom_mnist", output_dir="custom_mnist2")