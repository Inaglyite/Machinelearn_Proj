# mnist_draw_predict_tkinter_final.py
import tkinter as tk
from tkinter import messagebox
import torch
import torch.nn as nn
from PIL import ImageGrab, ImageOps, Image
from torchvision import transforms


# ====================== 1. 定义主应用类 ======================
class MNISTDrawApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST 手写数字识别")
        self.root.geometry("450x350")

        # 模型路径（需修改为实际路径）
        self.model_path = r"C:\Users\Lenovo\PycharmProjects\Machinelearn_Proj\model_pth\best_state_dict.pth"

        # 初始化界面
        self.init_ui()

        # 绘图相关变量
        self.last_x = None
        self.last_y = None
        self.line_width = 20  # 线条粗细

        # 加载模型
        self.model = None
        self.load_model()

    # ====================== 2. 界面初始化 ======================
    def init_ui(self):
        # 左侧画布区域
        self.canvas = tk.Canvas(
            self.root,
            width=280,
            height=280,
            bg="black",
            highlightthickness=0
        )
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        # 右侧控制面板
        control_frame = tk.Frame(self.root)
        control_frame.grid(row=0, column=1, sticky="n")

        # 按钮
        self.btn_clear = tk.Button(
            control_frame,
            text="清空画布 (C)",
            width=15,
            command=self.clear_canvas
        )
        self.btn_clear.pack(pady=5)

        self.btn_predict = tk.Button(
            control_frame,
            text="预测数字 (P)",
            width=15,
            command=self.predict_digit
        )
        self.btn_predict.pack(pady=5)

        # 结果显示
        self.result_label = tk.Label(
            control_frame,
            text="预测结果: ---",
            font=("Arial", 14),
            width=15
        )
        self.result_label.pack(pady=20)

        # 绑定事件
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset_position)
        self.root.bind("c", lambda e: self.clear_canvas())
        self.root.bind("p", lambda e: self.predict_digit())

    # ====================== 3. 核心功能方法 ======================
    def draw(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_line(
                self.last_x, self.last_y,
                event.x, event.y,
                width=self.line_width,
                fill="white",
                capstyle=tk.ROUND,
                smooth=True
            )
        self.last_x = event.x
        self.last_y = event.y

    def reset_position(self, event):
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.result_label.config(text="预测结果: ---")

    def get_canvas_image(self):
        """精确截取画布内容并返回PIL图像"""
        x = self.root.winfo_rootx() + self.canvas.winfo_x() + 2
        y = self.root.winfo_rooty() + self.canvas.winfo_y() + 2
        x_end = x + self.canvas.winfo_width() - 4
        y_end = y + self.canvas.winfo_height() - 4
        return ImageGrab.grab((x, y, x_end, y_end))

    def preprocess_image(self, img):
        """图像预处理（与训练时完全一致）"""
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
           # transforms.Lambda(lambda x: 1.0 - x)  # 颜色反转（关键步骤）
        ])
        return transform(img).unsqueeze(0)  # [1,1,28,28]

    def load_model(self):
        """加载与训练时结构完全一致的CNN模型"""
        try:
            # 定义与训练代码完全相同的模型结构
            class SimpleCNN(nn.Module):
                def __init__(self, num_class=10):
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(1, 16, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(16, 32, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                    )
                    self.classifier = nn.Sequential(
                        nn.Linear(32 * 7 * 7, 128),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(128, num_class),
                    )

                def forward(self, x):
                    x = self.features(x)
                    x = x.view(x.size(0), -1)
                    x = self.classifier(x)
                    return x

            # 加载模型参数
            self.model = SimpleCNN(num_class=10)
            self.model.load_state_dict(
                torch.load(self.model_path, map_location="cpu")
            )
            self.model.eval()
        except Exception as e:
            messagebox.showerror("致命错误",
                                 f"模型加载失败:\n{str(e)}\n"
                                 "请检查：\n1. 模型路径是否正确\n2. 模型结构是否匹配")
            self.root.destroy()

    def predict_digit(self):
        """执行预测"""
        try:
            # 获取并预处理图像
            pil_img = self.get_canvas_image()
            input_tensor = self.preprocess_image(pil_img)

            # 执行预测
            with torch.no_grad():
                outputs = self.model(input_tensor)
                prob, pred = torch.max(outputs.softmax(dim=1), 1)

            # 显示结果
            self.result_label.config(
                text=f"预测结果: {pred.item()}\n置信度: {prob.item():.2%}"
            )
        except Exception as e:
            messagebox.showerror("预测错误", f"预测失败:\n{str(e)}")


# ====================== 4. 启动程序 ======================
if __name__ == "__main__":
    root = tk.Tk()
    app = MNISTDrawApp(root)
    root.mainloop()