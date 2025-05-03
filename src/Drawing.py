# 这个文件的作用是制作一个APP，用手写板来绘制手写数字
# 要求如下：
## 1.在绘图软件中创建 28x28像素 的画布。
## 2.使用黑色画笔（RGB=0,0,0）在白色背景（RGB=255,255,255）上书写数字。
## 3.保存为 PNG 格式，命名格式为 数字_序号.png（如 5_001.png）

import tkinter as tk
from PIL import Image, ImageDraw
import os

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("这是一个手写数字绘制工具")
        self.canvas = tk.Canvas(self.root, width=280, height=280, bg="white")# 28*28的画布
        self.canvas.pack()
        self.image = Image.new("L", (280, 280), 255) #灰度模式
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)#绑定鼠标事件

        self.save_button = tk.Button(root, text="保存", command=self.save)
        self.save_button.pack()

        self.label_entry = tk.Entry(root)
        self.label_entry.pack()
        self.label_entry.insert(0, "请输入0-9标签")


    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-8, y-8, x+8, y+8, fill='black', outline='black')
        self.draw.ellipse([x-8, y-8, x+8, y+8], fill=0)

    def save(self):
        label = self.label_entry.get()
        if not label.isdigit() or int(label) < 0 or int(label) > 9:
            print("标签是0-9的整数")
            return
        os.makedirs('custom_mnist', exist_ok=True)
        count = len([f for f in os.listdir("custom_mnist") if f.startswith(label)])
        filename = f"custom_mnist/{label}_{count+1:03d}.png"
        self.image.resize((28,28)).save(filename)
        print(f"保存成功:{filename}")
        self.image = Image.new("L",(280,280), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.delete('all')

root = tk.Tk()
app = DrawingApp(root)
root.mainloop()

