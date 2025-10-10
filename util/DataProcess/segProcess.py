import torch
import numpy as np
import math
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import functional as TF
import random
from PIL import Image


# 该模块的方法都是基于输入语义分割模型前的处理和模型输出的后处理！！！


#=============================================================================
# 该方式为直接缩放
# 可能会破坏原始长宽比，导致物体变形（如拉伸或压缩）
# 适合场景：图像长宽比差异较小，或任务对物体形状变形不敏感
#=============================================================================
def resize_image_and_mask(image, mask, target_size=(512, 512)):
    """
    将图像和对应的 mask 缩放到固定大小，并对图像进行标准化。
    支持 PIL.Image 或 torch.Tensor 作为输入。

    参数:
        image (PIL.Image 或 torch.Tensor): 输入的 RGB 图像。
        mask (PIL.Image 或 torch.Tensor): 输入的 mask 标签。
        target_size (tuple): 缩放目标尺寸 (高度, 宽度)，如 (512, 512)。

    返回:
        torch.Tensor: 缩放且标准化后的图像张量 (3, H, W)。
        torch.Tensor: 缩放后的 mask 张量 (H, W)。
    """
    # 如果图像是 Tensor，转换为 PIL.Image
    if isinstance(image, torch.Tensor):
        image = T.ToPILImage()(image)

    # 如果 mask 是 Tensor，转换为 PIL.Image
    if isinstance(mask, torch.Tensor):
        mask = Image.fromarray(mask.numpy().astype(np.uint8))

    # 定义用于图像的 Resize 转换，使用双线性插值
    image_resize_transform = T.Resize(target_size, interpolation=T.InterpolationMode.BILINEAR)

    # 定义用于 mask 的 Resize 转换，使用最近邻插值
    mask_resize_transform = T.Resize(target_size, interpolation=T.InterpolationMode.NEAREST)

    # 应用 Resize 转换
    resized_image = image_resize_transform(image)
    resized_mask = mask_resize_transform(mask)

    # 定义标准化操作，使用 ImageNet 的均值和标准差
    normalization_transform = T.Compose([
        T.ToTensor(),  # 转换为张量
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    # 将图像应用标准化
    normalized_image_tensor = normalization_transform(resized_image)

    # 将 mask 转换为 numpy 数组并转为 PyTorch 的 long 类型张量（类别 ID 必须为整数）
    mask_tensor = torch.as_tensor(np.array(resized_mask), dtype=torch.long)

    return normalized_image_tensor, mask_tensor



#==========================================================================================
# 保持原始长款比，将图片缩放到目标尺寸的短边；在长边两侧填充零或边缘像素（如黑色或反射填充）以达到目标尺寸
# 该方式保留物体原始比例，避免形变
# 但填充区域可能引入无效信息，增加计算量
# 适用场景：物体长宽比较敏感（如文字、医学图像）
# YOLO系列算法貌似就是采用类似策略
#===========================================================================================
def resize_with_padding(image, mask, target_size=(512, 512), padding_mode="constant", pad_value=0):
    """
    保持图像和 mask 的长宽比，通过填充将其调整为目标尺寸。

    参数:
        image (PIL.Image 或 torch.Tensor): 输入的 RGB 图像。
        mask (PIL.Image 或 torch.Tensor): 输入的 mask 标签。
        target_size (tuple): 目标尺寸 (高度, 宽度)，如 (512, 512)。
        padding_mode (str): 填充模式，可以是 "constant"、"reflect"、"edge"。
        pad_value (int): 填充的常量值（仅在 constant 模式下使用）。

    返回:
        torch.Tensor: 填充调整后的图像张量 (3, H, W)。
        torch.Tensor: 填充调整后的 mask 张量 (H, W)。
    """
    # 如果图像是 Tensor，转换为 PIL.Image
    if isinstance(image, torch.Tensor):
        image = T.ToPILImage()(image)

    # 如果 mask 是 Tensor，转换为 PIL.Image
    if isinstance(mask, torch.Tensor):
        mask = Image.fromarray(mask.numpy().astype(np.uint8))

    # 获取目标宽高
    target_height, target_width = target_size

    # 原始图像尺寸
    original_width, original_height = image.size

    # 计算缩放比例，保持长宽比
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize 图像和 mask，保持长宽比
    image_resize_transform = T.Resize((new_height, new_width), interpolation=T.InterpolationMode.BILINEAR)
    mask_resize_transform = T.Resize((new_height, new_width), interpolation=T.InterpolationMode.NEAREST)

    resized_image = image_resize_transform(image)
    resized_mask = mask_resize_transform(mask)

    # 计算需要填充的尺寸
    pad_top = (target_height - new_height) // 2
    pad_bottom = target_height - new_height - pad_top
    pad_left = (target_width - new_width) // 2
    pad_right = target_width - new_width - pad_left

    # 定义填充转换
    padding_transform_image = T.Pad((pad_left, pad_top, pad_right, pad_bottom), fill=pad_value, padding_mode=padding_mode)
    padding_transform_mask = T.Pad((pad_left, pad_top, pad_right, pad_bottom), fill=0, padding_mode="constant")

    # 对图像和 mask 进行填充
    padded_image = padding_transform_image(resized_image)
    padded_mask = padding_transform_mask(resized_mask)

    # 转换为张量
    image_tensor = T.ToTensor()(padded_image)
    mask_tensor = torch.as_tensor(np.array(padded_mask), dtype=torch.long)

    return image_tensor, mask_tensor




#==================================================================
# 中心裁剪，可以截取图像中心区域
# 但是可能会丢失关键信息（如边缘物体）
# 适用场景：物体主要位于图像中心（如人脸识别）
#==================================================================
def center_crop(image, mask, crop_size=(512, 512)):
    """
    对图像和 mask 进行中心裁剪。

    参数:
        image (PIL.Image 或 torch.Tensor): 输入的 RGB 图像。
        mask (PIL.Image 或 torch.Tensor): 输入的 mask 标签。
        crop_size (tuple): 裁剪的目标尺寸 (高度, 宽度)。

    返回:
        torch.Tensor: 中心裁剪后的图像张量 (3, H, W)。
        torch.Tensor: 中心裁剪后的 mask 张量 (H, W)。
    """
    # 如果图像是 Tensor，转换为 PIL.Image
    if isinstance(image, torch.Tensor):
        image = T.ToPILImage()(image)

    # 如果 mask 是 Tensor，转换为 PIL.Image
    if isinstance(mask, torch.Tensor):
        mask = Image.fromarray(mask.numpy().astype(np.uint8))

    # 定义中心裁剪
    center_crop_transform = T.CenterCrop(crop_size)

    # 应用裁剪
    cropped_image = center_crop_transform(image)
    cropped_mask = center_crop_transform(mask)

    # 转换为张量
    image_tensor = T.ToTensor()(cropped_image)
    mask_tensor = torch.as_tensor(np.array(cropped_mask), dtype=torch.long)

    return image_tensor, mask_tensor




#=============================================================================
# 该方法为随机裁剪，随机截取子区域，在训练时增强多样性
# 可能会丢失关键信息
#=============================================================================
def random_crop(image, mask, crop_size=(256, 256)):
    """
    对图像和 mask 进行随机裁剪，保证裁剪区域相同。

    参数:
        image (PIL.Image 或 torch.Tensor): 输入的 RGB 图像。
        mask (PIL.Image 或 torch.Tensor): 输入的 mask 标签。
        crop_size (tuple): 裁剪的目标尺寸 (高度, 宽度)。

    返回:
        torch.Tensor: 随机裁剪后的图像张量 (3, H, W)。
        torch.Tensor: 随机裁剪后的 mask 张量 (H, W)。
    """
    # 如果图像是 Tensor，转换为 PIL.Image
    if isinstance(image, torch.Tensor):
        image = T.ToPILImage()(image)

    # 如果 mask 是 Tensor，转换为 PIL.Image
    if isinstance(mask, torch.Tensor):
        mask = Image.fromarray(mask.numpy().astype(np.uint8))

    # 获取图像的原始尺寸
    width, height = image.size
    crop_height, crop_width = crop_size

    # 随机生成裁剪左上角的坐标 (top, left)
    if height > crop_height:
        top = random.randint(0, height - crop_height)
    else:
        top = 0
    if width > crop_width:
        left = random.randint(0, width - crop_width)
    else:
        left = 0

    # 裁剪区域的右下角坐标
    right = left + crop_width
    bottom = top + crop_height

    # 裁剪图像和 mask，确保使用相同的区域
    cropped_image = image.crop((left, top, right, bottom))
    cropped_mask = mask.crop((left, top, right, bottom))

    # 转换为张量
    image_tensor = T.ToTensor()(cropped_image)
    mask_tensor = torch.as_tensor(np.array(cropped_mask), dtype=torch.long)

    return image_tensor, mask_tensor


#==================================================================
# Resize+Crop的多尺度训练策略，可提升模型对不同尺度的鲁棒性
# 但是训练复杂度增加
# 适用场景：数据集中物体尺度差异较大（如遥感图像）
#==================================================================
def multi_scale_resize_crop(image, mask, crop_size=(512, 512), scales=[0.5, 1.0, 1.5, 2.0]):
    """
    多尺度训练策略，从给定的尺度中随机选择一个进行 Resize + Crop 操作。

    参数:
        image (PIL.Image 或 torch.Tensor): 输入的 RGB 图像。
        mask (PIL.Image 或 torch.Tensor): 输入的 mask 标签。
        crop_size (tuple): 裁剪的目标尺寸 (高度, 宽度)。
        scales (list): 多尺度训练的缩放比例列表。

    返回:
        torch.Tensor: 经过随机尺度 Resize + Crop 处理后的图像张量。
        torch.Tensor: 经过随机尺度 Resize + Crop 处理后的 mask 张量。
    """
    # 如果输入为 Tensor，转换为 PIL.Image
    if isinstance(image, torch.Tensor):
        image = T.ToPILImage()(image)
    if isinstance(mask, torch.Tensor):
        mask = Image.fromarray(mask.numpy().astype(np.uint8))

    # Step 1: 随机选择一个尺度
    selected_scale = random.choice(scales)
    print(f"选择的随机尺度: {selected_scale}")

    # Step 2: Resize 图像和 mask
    original_width, original_height = image.size
    new_width = int(original_width * selected_scale)
    new_height = int(original_height * selected_scale)

    resize_transform_image = T.Resize((new_height, new_width), interpolation=T.InterpolationMode.BILINEAR)
    resize_transform_mask = T.Resize((new_height, new_width), interpolation=T.InterpolationMode.NEAREST)

    resized_image = resize_transform_image(image)
    resized_mask = resize_transform_mask(mask)

    # Step 3: 调用自定义的随机裁剪函数
    image_tensor, mask_tensor = random_crop(resized_image, resized_mask, crop_size)

    return image_tensor, mask_tensor


#====================================================================
# Resize+Crop的多尺度训练策略，可提升模型对不同尺度的鲁棒性
# 但是训练复杂度增加
# 适用场景：数据集中物体尺度差异较大（如遥感图像）
#====================================================================
def multi_scale_resize_pad(image, mask, target_size=(512, 512), scales=[0.5, 1.0, 1.5, 2.0], padding_mode="constant", pad_value=0):
    """
    多尺度训练策略，随机从给定尺度中选择一个，并执行 Resize + Pad 操作。

    参数:
        image (PIL.Image 或 torch.Tensor): 输入的 RGB 图像。
        mask (PIL.Image 或 torch.Tensor): 输入的 mask 标签。
        target_size (tuple): 填充后的目标尺寸 (高度, 宽度)。
        scales (list): 多尺度训练的缩放比例列表。
        padding_mode (str): 填充模式，可以是 "constant"、"reflect"、"edge"。
        pad_value (int): 填充的常量值（仅在 constant 模式下使用）。

    返回:
        torch.Tensor: 经过随机尺度 Resize + Pad 处理后的图像张量。
        torch.Tensor: 经过随机尺度 Resize + Pad 处理后的 mask 张量。
    """
    # 如果输入为 Tensor，转换为 PIL.Image
    if isinstance(image, torch.Tensor):
        image = T.ToPILImage()(image)
    if isinstance(mask, torch.Tensor):
        mask = Image.fromarray(mask.numpy().astype(np.uint8))

    # Step 1: 随机选择一个尺度
    selected_scale = random.choice(scales)
    print(f"选择的随机尺度: {selected_scale}")

    # Step 2: 调用 resize_with_padding
    original_width, original_height = image.size
    new_width = int(original_width * selected_scale)
    new_height = int(original_height * selected_scale)

    # 调用之前定义的 resize_with_padding
    resized_padded_image, resized_padded_mask = resize_with_padding(
        image, mask, target_size=target_size, padding_mode=padding_mode, pad_value=pad_value
    )

    return resized_padded_image, resized_padded_mask


#-----------------------------------------------------------------------------
# 先Resize到一个较大尺度的图像，该尺度可以手动传入，在crop到正方形
# crop可选两种模式，随机裁剪和中心裁剪
#-----------------------------------------------------------------------------
def resize_and_crop(image, mask, resize_size=(600, 600), crop_size=(512, 512), crop_mode="random"):
    """
    Resize + Crop 组合策略：
    先将图像和 mask 缩放到较大尺寸，再根据参数选择随机裁剪或中心裁剪。

    参数:
        image (PIL.Image 或 torch.Tensor): 输入的 RGB 图像。
        mask (PIL.Image 或 torch.Tensor): 输入的 mask 标签。
        resize_size (tuple): 缩放后的尺寸 (高度, 宽度)。
        crop_size (tuple): 裁剪的目标尺寸 (高度, 宽度)。
        crop_mode (str): 裁剪模式，可选 "random" 或 "center"。

    返回:
        torch.Tensor: Resize + Crop 后的图像张量 (3, H, W)。
        torch.Tensor: Resize + Crop 后的 mask 张量 (H, W)。
    """
    # 如果输入是 Tensor，转换为 PIL.Image
    if isinstance(image, torch.Tensor):
        image = T.ToPILImage()(image)
    if isinstance(mask, torch.Tensor):
        mask = Image.fromarray(mask.numpy().astype(np.uint8))

    # Step 1: Resize 图像和 mask
    resize_transform_image = T.Resize(resize_size, interpolation=T.InterpolationMode.BILINEAR)
    resize_transform_mask = T.Resize(resize_size, interpolation=T.InterpolationMode.NEAREST)

    resized_image = resize_transform_image(image)
    resized_mask = resize_transform_mask(mask)

    # Step 2: 根据裁剪模式调用相应的裁剪函数
    if crop_mode == "random":
        image_tensor, mask_tensor = random_crop(resized_image, resized_mask, crop_size)
    elif crop_mode == "center":
        image_tensor, mask_tensor = center_crop(resized_image, resized_mask, crop_size)
    else:
        raise ValueError(f"Unsupported crop_mode: {crop_mode}. Use 'random' or 'center'.")

    return image_tensor, mask_tensor

