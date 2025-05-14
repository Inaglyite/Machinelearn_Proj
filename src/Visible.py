import numpy as np
import matplotlib.pyplot as plt
import os


def read_idx_images(img_file):
    """读取 IDX 图像文件"""
    with open(img_file, 'rb') as f:
        # 读取文件头
        magic = int.from_bytes(f.read(4), byteorder='big')
        num_images = int.from_bytes(f.read(4), byteorder='big')
        rows = int.from_bytes(f.read(4), byteorder='big')
        cols = int.from_bytes(f.read(4), byteorder='big')
        # 读取所有图像数据 [N, H, W]
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
    return images


def read_idx_labels(label_file):
    """读取 IDX 标签文件"""
    with open(label_file, 'rb') as f:
        magic = int.from_bytes(f.read(4), byteorder='big')
        num_labels = int.from_bytes(f.read(4), byteorder='big')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def visualize_idx_samples(img_file, label_file, num_samples=9, save_path=None):
    """可视化 IDX 数据集样本"""
    images = read_idx_images(img_file)
    labels = read_idx_labels(label_file)

    # 随机选择样本
    indices = np.random.choice(len(images), num_samples, replace=False)
    sample_images = images[indices]
    sample_labels = labels[indices]

    # 创建画布
    plt.figure(figsize=(12, 12))
    for i, (img, label) in enumerate(zip(sample_images, sample_labels)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Label: {label}", fontsize=14)
        plt.axis('off')

    # 保存或显示
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"可视化结果已保存至 {save_path}")
    else:
        plt.show()


# 使用示例 ---------------------------------------------------
if __name__ == "__main__":
    # 参数配置
    train_images = 'D:\ProjectSava\PythonProj\Machinelearn_Proj\Mydata\custom_mnist2-images-idx3-ubyte'  # 自定义数据集图像
    train_labels = 'D:\ProjectSava\PythonProj\Machinelearn_Proj\Mydata\custom_mnist2-labels-idx1-ubyte'  # 自定义数据集标签

    # 可视化训练集样本（保存到文件）
    visualize_idx_samples(
        img_file=train_images,
        label_file=train_labels,
        num_samples=9,
        save_path="./visualization/custom_dataset_samples.png"
    )

    # 可视化MNIST测试集样本对比（直接显示）
    mnist_test_images = 'D:\ProjectSava\PythonProj\Machinelearn_Proj\data\MNIST\\raw\\t10k-images-idx3-ubyte'
    mnist_test_labels = 'D:\ProjectSava\PythonProj\Machinelearn_Proj\data\MNIST\\raw\\t10k-labels-idx1-ubyte'
    visualize_idx_samples(mnist_test_images, mnist_test_labels)