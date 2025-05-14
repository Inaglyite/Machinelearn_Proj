# 这个文件的作用是部署Gradio
import gradio as gr
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from simplecnn import simplecnn  # 确保从你的模型文件中导入

# 加载训练好的模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = simplecnn(num_class=10).to(device)
model.load_state_dict(torch.load("model_pth/best.pth", map_location=device))
model.eval()

# 定义预处理流程
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def recognize_digit(image):
    # 转换输入图像
    if image is None:
        return ""

    # 转换格式并预处理
    image = Image.fromarray(image.astype('uint8')).convert('L')
    image = transform(image).unsqueeze(0).to(device)

    # 模型推理
    with torch.no_grad():
        output = model(image)
        prob = torch.nn.functional.softmax(output, dim=1)[0]
        pred = torch.argmax(prob).item()

    # 返回结果和置信度
    return f"预测数字：{pred}\n" + "\n".join([f"{i}: {prob[i].item() * 100:.1f}%" for i in range(10)])


# 创建Gradio界面
interface = gr.Interface(
    fn=recognize_digit,
    inputs=gr.Sketchpad(shape=(280, 280), brush_radius=10, image_mode="L"),
    outputs=gr.Textbox(label="识别结果"),
    title="手写数字识别",
    description="在左侧画布书写0-9数字，右侧显示识别结果",
    allow_flagging="never"
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)