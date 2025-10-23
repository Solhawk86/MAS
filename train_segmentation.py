import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

def l1_loss(pred, target):
    return F.l1_loss(pred, target)

def train_segmentation(segmentation_model,
                       train_dataset,
                       device='cuda',
                       batch_size=2,
                       lr=1e-4,
                       num_epochs=20,
                       output_dir='./output'):
    """
    segmentation_model: build_segmentation_decoder 返回的模型
    train_dataset: Dataset
    """
    os.makedirs(output_dir, exist_ok=True)

    # 冻结 backbone，只训练 head
    backbone = segmentation_model[0]
    decoder = segmentation_model[1]

    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    decoder.train()
    decoder.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        running_loss = 0.0

        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                features = backbone(imgs)  # backbone 输出特征

            pred = decoder(features)['pred_masks']  # Mask2FormerHead 输出 mask

            # 保证 pred 和 mask 尺寸一致
            if pred.shape[-2:] != masks.shape[-2:]:
                pred = F.interpolate(pred, size=masks.shape[-2:], mode='bilinear', align_corners=False)

            loss = l1_loss(pred, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"训练集 L1 Loss: {avg_loss:.4f}")

        # 保存每个 epoch 的模型
        torch.save(decoder.state_dict(), os.path.join(output_dir, f"head_epoch{epoch+1}.pth"))

    print("训练完成！")
