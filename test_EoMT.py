"""
EoMT 模型验证集推理脚本
---------------------------------------
功能：
1. 从 checkpoint (.pt 或 .ckpt) 加载模型
2. 遍历验证集的所有图像
3. 输出对应的 mask（黑白图）
   - 背景 = 0（黑）
   - 前景 = 255（白）
4. 输出保存为 JPG 格式

作者：Henry
"""

import os
import torch
from PIL import Image
from tqdm import tqdm
import lightly_train
import matplotlib.pyplot as plt
import torch
from torchvision.io import read_image
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms.functional import to_pil_image


# ============================

# 训练好的 EoMT 模型权重（.pt 或 .ckpt）
MODEL_PATH = "/home/yangyan/hong/MAS/data/results/weights/2025.10.14/exported_models/exported_best.pt"

# 验证集图像文件夹
VAL_IMAGE_DIR = "/home/yangyan/hong/Dual_SAM/dataset/RMAS/test/img"

# 保存预测 mask 的文件夹
SAVE_MASK_DIR = "/home/yangyan/hong/MAS/data/results/weights/2025.10.14/results/RMAS"

# ============================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def main():
    # 1. 检查输出目录
    ensure_dir(SAVE_MASK_DIR)

    # 2. 加载模型
    print(f" Loading model from {MODEL_PATH} ...")
    model = lightly_train.load_model_from_checkpoint(MODEL_PATH)
    model.eval()

    # 3. 获取验证集图片
    image_files = sorted([
        f for f in os.listdir(VAL_IMAGE_DIR)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    print(f" Found {len(image_files)} validation images.")

    # 4. 遍历推理
    for img_name in tqdm(image_files, desc="Predicting masks"):
        img_path = os.path.join(VAL_IMAGE_DIR, img_name)

        # 推理
        with torch.no_grad():
            mask = model.predict(img_path)

        # mask 是类别ID张量（每像素为 0 或 255）
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()

        mask_img = Image.fromarray(mask.astype("uint8"))
        save_path = os.path.join(SAVE_MASK_DIR, os.path.splitext(img_name)[0] + ".jpg")
        mask_img.save(save_path, "JPEG")

    print(f"Inference complete.")

# def main():
#     model = lightly_train.load_model_from_checkpoint(
#     MODEL_PATH
#     )
#     model.eval()    
#     image = read_image("/home/mil201/hong/MAS/data/dataset/MAS3K/val/images/MAS_Arthropod_Crab_Cam_362.jpg")
#     masks = model.predict(image)

#     masks = torch.stack([masks == class_id for class_id in masks.unique()])
#     image_with_masks = draw_segmentation_masks(image, masks, alpha=0.6)
#     image_with_masks = to_pil_image(image_with_masks)
#     image_with_masks.save("/home/mil201/hong/MAS/data/results/MAS3K_with_EoMT/MAS_Arthropod_Crab_Cam_362_with_masks.jpg")

if __name__ == "__main__":
    main()
