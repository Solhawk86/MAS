from PIL import Image
import os

gt_path = "/home/mil201/hong/MAS/data/dataset/MAS3K/train/masks/MAS_Mollusc_StarFish_Com_266.png"
pred_path = "/home/mil201/hong/MAS/data/dataset/MAS3K/train/images/MAS_Mollusc_StarFish_Com_267.jpg"

gt = Image.open(gt_path)
pred = Image.open(pred_path)

# 将 GT resize 到预测图同尺寸
gt_resized = gt.resize(pred.size, Image.NEAREST)  # 对mask用最近邻
gt_resized.save("/home/mil201/hong/MAS/data/dataset/MAS3K/train/masks/MAS_Arthropod_Shrimp_Com_537.png")
