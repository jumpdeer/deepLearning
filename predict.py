import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

# =======================
# 配置参数
# =======================
MODEL_PATH = "path_to_your_model.pth"  # 模型权重文件路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 使用 GPU 或 CPU
NUM_CLASSES = 20  # CIHP 数据集类别数
INPUT_IMAGE = "path_to_input_image.jpg"  # 输入图像路径
OUTPUT_MASK = "output_mask.png"  # 输出掩码图像路径

# CIHP 类别颜色映射
CIHP_COLORS = [
    [0, 0, 0], [128, 0, 0], [255, 0, 0], [0, 85, 0], [170, 0, 51],
    [255, 85, 0], [0, 0, 85], [0, 119, 221], [85, 85, 0], [0, 85, 85],
    [85, 51, 0], [52, 86, 128], [0, 128, 0], [0, 0, 255], [51, 170, 221],
    [0, 255, 255], [85, 255, 170], [170, 255, 85], [255, 255, 0], [255, 170, 0]
]

# =======================
# 加载模型
# =======================
def load_model(model_path):
    # 假设使用 DeepLabv3+，根据实际模型架构替换
    from torchvision.models.segmentation import deeplabv3_resnet50
    model = deeplabv3_resnet50(pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()  # 评估模式
    return model

# =======================
# 图像预处理
# =======================
def preprocess_image(image_path, input_size=(512, 512)):
    img = Image.open(image_path).convert("RGB")
    original_size = img.size  # 保存原始尺寸

    preprocess = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img).unsqueeze(0)  # 增加 batch 维度
    return img_tensor, original_size

# =======================
# 预测函数
# =======================
def predict(model, input_tensor, original_size):
    with torch.no_grad():
        input_tensor = input_tensor.to(DEVICE)
        output = model(input_tensor)["out"]  # 获取模型输出
        output = torch.argmax(output.squeeze(), dim=0).cpu().numpy()  # 最大类别索引
    output_resized = cv2.resize(output, original_size, interpolation=cv2.INTER_NEAREST)  # 调整回原始大小
    return output_resized

# =======================
# 保存掩码图像
# =======================
def save_mask(mask, output_path, colors=CIHP_COLORS):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(colors):
        color_mask[mask == class_id] = color
    cv2.imwrite(output_path, color_mask[:, :, ::-1])  # 转换为 BGR 格式以适配 OpenCV

# =======================
# 主流程
# =======================
if __name__ == "__main__":
    # 1. 加载模型
    model = load_model(MODEL_PATH)

    # 2. 图像预处理
    input_tensor, original_size = preprocess_image(INPUT_IMAGE)

    # 3. 预测
    mask = predict(model, input_tensor, original_size)

    # 4. 保存结果
    save_mask(mask, OUTPUT_MASK)
    print(f"Prediction completed. Mask saved to {OUTPUT_MASK}.")
