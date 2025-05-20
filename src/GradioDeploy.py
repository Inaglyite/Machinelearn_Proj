import gradio as gr
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms


# ====================== 1. 模型定义与加载 ======================
class SimpleCNN(nn.Module):
    def __init__(self, num_class=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_class))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


model = SimpleCNN()
model.load_state_dict(torch.load(
    r"C:\Users\Lenovo\PycharmProjects\Machinelearn_Proj\model_pth\best_state_dict.pth",
    map_location="cpu"
))
model.eval()

# ====================== 2. 图像预处理管道 ======================
preprocess = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# ====================== 3. 预测函数 ======================
def predict_digit(img):
    if img is None:
        return {"预测结果": "---", "置信度": "---"}

    try:
        # 关键修改：手动颜色反转
        img = 255 - np.array(img)  # 白底黑字转黑底白字
        pil_img = Image.fromarray(img).convert("L")  # 确保灰度

        # 调试用：保存中间图像
        # pil_img.save("debug_input.png")

        input_tensor = preprocess(pil_img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            prob, pred = torch.max(outputs.softmax(dim=1), 1)

        return {
            "预测结果": str(pred.item()),
            "置信度": f"{prob.item():.2%}"
        }
    except Exception as e:
        return {"预测结果": "错误", "置信度": str(e)}


# ====================== 4. 构建Gradio界面 ======================
with gr.Blocks(title="MNIST识别", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 MNIST手写数字识别系统")

    with gr.Row():
        # 绘图区域（关键配置）
        with gr.Column():
            canvas = gr.Image(
                label="绘制区域",
                height=280,
                width=280,
                image_mode="L",
                tool="sketch",
                brush_radius=12,
                show_download_button=False,
                invert_colors=True  # 确保画布显示白底黑字
            )

        # 结果展示
        with gr.Column():
            output = gr.Label(
                label="识别结果",
                value={"预测结果": "---", "置信度": "---"},
                num_top_classes=1
            )

    # 控制按钮
    with gr.Row():
        clear_btn = gr.Button("清空画布", variant="secondary")
        predict_btn = gr.Button("开始识别", variant="primary")

    # 事件绑定
    predict_btn.click(
        fn=predict_digit,
        inputs=canvas,
        outputs=output
    )

    clear_btn.click(
        fn=lambda: [None, {"预测结果": "---", "置信度": "---"}],
        outputs=[canvas, output]
    )

# ====================== 5. 启动应用 ======================
if __name__ == "__main__":
    demo.launch(server_port=7860)