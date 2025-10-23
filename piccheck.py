import os
from PIL import Image
import numpy as np

def check_and_fix_resolution(pred_dir, gt_dir, pred_suffix=".jpg", gt_suffix=".png"):
    """
    检查预测图和GT的分辨率是否一致，如果不一致，将GT resize到预测图尺寸。

    参数:
        pred_dir (str): 预测图所在文件夹路径
        gt_dir (str): GT图所在文件夹路径
        pred_suffix (str): 预测图的文件后缀名（默认 .jpg）
        gt_suffix (str): GT图的文件后缀名（默认 .png）
    """
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(pred_suffix)])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(gt_suffix)])

    if len(pred_files) != len(gt_files):
        print(f"文件数量不一致：pred = {len(pred_files)}, gt = {len(gt_files)}")

    mismatch_count = 0

    for pred_name, gt_name in zip(pred_files, gt_files):
        pred_path = os.path.join(pred_dir, pred_name)
        gt_path = os.path.join(gt_dir, gt_name)

        try:
            pred_img = Image.open(pred_path)
            gt_img = Image.open(gt_path)

            if pred_img.size != gt_img.size:
                mismatch_count += 1
                print(f"分辨率不匹配: {pred_name} / {gt_name}")
                print(f"pred: {pred_img.size}, gt: {gt_img.size}")

                # resize GT 到预测图大小
                gt_resized = gt_img.resize(pred_img.size, Image.NEAREST)
                gt_resized.save(gt_path)
                print(f"  已将 GT resize 到 {pred_img.size}")

        except Exception as e:
            print(f"⚠️ 无法读取 {pred_name} 或 {gt_name}: {e}")

    if mismatch_count == 0:
        print("所有图像分辨率一致！")
    else:
        print(f"检查完成，共发现 {mismatch_count} 个不匹配的图像，并已修正。")
        
def value_check(path):
    '''
    
    检查 mask 图像的像素值范围和唯一值。
    分割任务中，0类别是背景，1或255类别是前景。
    '''
    mask_path = path
    mask = Image.open(mask_path).convert("L")
    mask_array = np.array(mask)
    print("mask形状:", mask_array.shape)
    print("数据类型:", mask_array.dtype)
    print("像素最小值:", mask_array.min())
    print("像素最大值:", mask_array.max())
    print("唯一像素值:", np.unique(mask_array))

if __name__ == "__main__":
    value_check("/home/yangyan/hong/MAS/data/dataset/UFO-120/TEST/mask/set_f0.jpg")
