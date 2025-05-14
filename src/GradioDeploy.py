# 这个文件的作用是部署Gradio
import gradio as gr
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from CNN import simplecnn  # 确保从你的模型文件中导入

# 加载训练好的模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = simplecnn(num_class=10).to(device)

# 修正后的模型路径（使用原始字符串）
model_path = r"D:\ProjectSava\PythonProj\Machinelearn_Proj\model_pth\best.pth"
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("模型加载成功")
except Exception as e:
    print(f"模型加载失败: {e}")

# 定义预处理流程
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 确保图片大小为28x28
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1.0 - x),  # MNIST数据集的反色处理
    transforms.Normalize((0.1307,), (0.3081,))
])


def recognize_digit(image_data):
    print("Debug: 输入数据结构 =", image_data.keys() if image_data else "None")

    if not image_data or "composite" not in image_data or image_data["composite"] is None:
        return "请绘制数字"

    try:
        composite_data = image_data["composite"]
        print("Debug: composite 数据类型 =", type(composite_data))
        print("Debug: composite 数据 =", composite_data)

        # 检查 composite 数据是否有效
        if composite_data is None:
            return "绘制的图像数据为空，请重新绘制"

        # 保存 composite 数据为图像
        composite_array = np.array(composite_data).astype('uint8')
        Image.fromarray(composite_array).save("debug_composite_image.png")
        print("Debug: composite 图像已保存为 debug_composite_image.png")

        # 转换为 NumPy 数组
        image_array = np.array(composite_data).astype('uint8')
        print("Debug: 图像数组形状 =", image_array.shape)

        # 转换为PIL图像并预处理
        image = Image.fromarray(image_array).convert('L')
        print("Debug: PIL图像大小 =", image.size)

        # 保存原始输入图像
        image.save("debug_original_input.png")
        print("Debug: 原始输入图像已保存为 debug_original_input.png")

        # 预处理图像
        image = transform(image).unsqueeze(0).to(device)
        print("Debug: 预处理后的图像张量形状 =", image.shape)

        # 保存预处理后的图像
        from torchvision.utils import save_image
        save_image(image, "debug_preprocessed_image.png")
        print("Debug: 预处理后的图像已保存为 debug_preprocessed_image.png")

        # 模型推理
        with torch.no_grad():
            output = model(image)
            print("Debug: 模型原始输出 =", output)

            prob = torch.nn.functional.softmax(output, dim=1)[0]
            print("Debug: Softmax 概率 =", prob)

            pred = torch.argmax(prob).item()
            print("Debug: 预测结果 =", pred)

        result = [f"预测数字：{pred}（置信度：{prob[pred].item() * 100:.1f}%）"]
        result += [f"{i}: {prob[i].item() * 100:.1f}%" for i in range(10)]
        return "\n".join(result)
    except Exception as e:
        print(f"发生错误: {e}")
        return f"发生错误：{str(e)}"

# 创建Gradio界面
interface = gr.Interface(
    fn=recognize_digit,
    inputs=gr.Sketchpad(
        height=280,
        width=280,
        brush=gr.Brush(colors=["black"]),
        image_mode="L"
    ),
    outputs=gr.Textbox(label="识别结果"),
    title="手写数字识别",
    description="在左侧画布书写0-9数字，右侧显示识别结果",
    flagging_mode="never"
)

if __name__ == "__main__":
    print("启动 Gradio 应用...")
    interface.launch(share=True)