import torch
import yaml
from MAS.utils import *
from model import DualImageDINOv3
from data.data import *
from torch.utils.data import DataLoader,RandomSampler
from loss import *

# ===  读取 YAML 文件 ===
cfg = load_config('config_eval.yaml')
    
device = "cuda" if torch.cuda.is_available() else "cpu"

# === 构建模型 ===
model = DualImageDINOv3(cfg=cfg, device=device)
model.eval()

# === 输入 ===
raw_input = Dataset(
    base_dir=cfg['input_images']['paths'],
    dataset_name=cfg['input_images']['dataset_name'],
    transform= None
)
enhance_input = Dataset(
    base_dir=cfg['input_images']['paths'],
    dataset_name=cfg['input_images']['dataset_name'],
    transform= make_enhance_transform()
)

# === 采样策略 ===
sampler1 = RandomSampler(raw_input)

dual_loader = InfiniteDualDataloader(
    raw_input,
    enhance_input,
    batch_size= cfg['train']['batchsize'],
    max_steps=cfg['train']['max_steps'],
)


for raw_batch, enhance_batch in dual_loader:
    feat1, feat2 = model(raw_batch, enhance_batch)
    loss = self_supervised_loss(feat1, feat2)
    loss.backward()
