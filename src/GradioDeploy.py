# 这个文件的作用是部署Gradio
import gradio as gr
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageEnhance
from CNN import simplecnn  # 确保从你的模型文件中导入
from torchvision.utils import save_image

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
    transforms.Resize((28, 28)),  # 调整大小为28x28
    transforms.ToTensor(),  # 转为Tensor（范围[0,1]）
])



# --------------------- 核心推理函数 ---------------------
def recognize_digit(image_data):
    # 输入数据完整性检查
    if not image_data or "composite" not in image_data:
        return "请绘制数字"

    composite_data = image_data["composite"]
    if composite_data is None:
        return "绘制的图像数据为空，请重新绘制"

    # 转换为NumPy数组并检查是否全黑
    image_array = np.array(composite_data).astype('uint8')

    try:
        # 保存原始输入图像（调试用）
        Image.fromarray(image_array).save("debug_original_input.png")

        # 加载原始输入图像
        image_path = "debug_original_input.png"
        original_image = Image.open(image_path).convert('RGBA')
        image_array = np.array(original_image)

        # 分离 RGB 和 Alpha 通道
        r, g, b, a = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2], image_array[:, :, 3]

        # 替换透明区域为白色
        background_mask = (a == 0)  # 透明区域
        r[background_mask] = 255  # 红色通道填充为白色
        g[background_mask] = 255  # 绿色通道填充为白色
        b[background_mask] = 255  # 蓝色通道填充为白色
        a[background_mask] = 255  # 填充为不透明

        # 合并回 RGBA 图像
        new_image_array = np.dstack((r, g, b, a))
        new_image = Image.fromarray(new_image_array.astype(np.uint8))

        # 转换为灰度图像
        gray_image = new_image.convert('L')

        # 增强对比度
        enhancer = ImageEnhance.Contrast(gray_image)
        enhanced_image = enhancer.enhance(2.0)  # 增强因子为2.0（可以根据需要调整）

        # 继续预处理
        image = transform(enhanced_image).unsqueeze(0)

        # 打印预处理后的张量值范围
        print("Preprocessed image tensor min:", image.min().item())
        print("Preprocessed image tensor max:", image.max().item())

        # 保存预处理后的图像
        save_image(image, "debug_preprocessed_image.png")

        # 模型推理
        with torch.no_grad():
            output = model(image)
            prob = torch.nn.functional.softmax(output, dim=1)[0]
            pred = torch.argmax(prob).item()

        # 格式化结果
        result = [f"预测数字：{pred}（置信度：{prob[pred].item() * 100:.1f}%）"]
        result += [f"{i}: {prob[i].item() * 100:.1f}%" for i in range(10)]
        return "\n".join(result)
    except Exception as e:
        return f"发生错误：{str(e)}"


# --------------------- Gradio界面配置 ---------------------
interface = gr.Interface(
    fn=recognize_digit,
    inputs=gr.Sketchpad(
        height=280,
        width=280,
        brush=gr.Brush(colors=["black"])  # 参数名必须为colors
    ),  # 确保括号闭合
    outputs=gr.Textbox(label="识别结果"),
    title="手写数字识别",
    description="在左侧画布书写0-9数字，右侧显示识别结果",
    allow_flagging="never"
)

# --------------------- 启动应用 ---------------------
if __name__ == "__main__":
    print("启动Gradio应用...")
    interface.launch(share=True)