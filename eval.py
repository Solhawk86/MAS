import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# ------------------------------
# 用户需要填写的路径
# ------------------------------
def cal_iou():
    class IoUCalculator:
        def __init__(self):
            self.total_iou = 0.0
            self.count = 0

        def update(self, pred: np.ndarray, gt: np.ndarray):
            """
            计算单张图的 IoU 并更新总和。
            pred 和 gt 都是二值化后的 numpy 数组，shape (H, W)，值为 0 或 1。
            """
            intersection = np.logical_and(pred, gt).sum()
            union = np.logical_or(pred, gt).sum()
            iou = intersection / union if union != 0 else 1.0

            self.total_iou += iou
            self.count += 1
            return iou

        def show(self):
            """
            返回当前的平均 IoU。
            """
            return self.total_iou / self.count if self.count != 0 else 0.0

    return IoUCalculator()

pred_folder = "/home/mil201/hong/MAS/data/results/MAS3K"   # 预测图文件夹
gt_folder   = "/home/mil201/hong/MAS/data/dataset/MAS3K/test/Mask"   # GT 文件夹

# ------------------------------
# 自动匹配文件（同名但不同后缀）
# ------------------------------
pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith('.jpg')])
gt_files   = sorted([f for f in os.listdir(gt_folder) if f.endswith('.png')])

# 确保文件名（去掉扩展名后）匹配
pred_names = [os.path.splitext(f)[0] for f in pred_files]
gt_names   = [os.path.splitext(f)[0] for f in gt_files]
common_names = sorted(list(set(pred_names) & set(gt_names)))

assert len(common_names) > 0, "未找到匹配的文件名，请检查预测图和GT文件夹。"

# ------------------------------
# 初始化 IoU 计算器
# ------------------------------
iou_calculator = cal_iou()

# ------------------------------
# 遍历文件计算 IoU
# ------------------------------
for name in tqdm(common_names, desc="Calculating IoU", ncols=80):
    pred_path = os.path.join(pred_folder, name + ".jpg")
    gt_path   = os.path.join(gt_folder, name + ".png")

    if not os.path.exists(pred_path) or not os.path.exists(gt_path):
        print(f"⚠️ 文件缺失: {name}")
        continue

    pred_img = np.array(Image.open(pred_path).convert("L")) / 255.0
    gt_img   = np.array(Image.open(gt_path).convert("L")) / 255.0

    assert pred_img.shape == gt_img.shape, f"{name} 分辨率不一致！"

    iou = iou_calculator.update(pred_img, gt_img)
    # tqdm.write(f"{name}: IoU = {iou:.4f}")

# ------------------------------
# 输出平均 IoU
# ------------------------------
mean_iou = iou_calculator.show()
print("\n============================")
print(f"✅ Mean IoU (mIoU): {mean_iou:.4f}")
print("============================")