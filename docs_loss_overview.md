# DINOv3-LoRA + Mask2Former 分割训练管线说明

## 1. Pipeline 概览
本文档对应 `MAS/train_m2f.py` 的完整训练流程：在冻结（或局部注入 LoRA 的）DINOv3 backbone 上，训练 HuggingFace Mask2Former 语义分割头，并辅以深度一致性正则。训练脚本通过 YAML 配置统一描述数据、模型、优化器、调度器与评估策略，最终输出最佳 checkpoint 及验证集预测结果，便于后续写作 CCF-A 类会议论文时引用。

核心阶段如下：
- **数据阶段**：`build_segmentation_image_dataloaders_from_cfg` 读取数据根目录、增强和采样方式，返回分布式感知的 train/val dataloader。
- **特征提取**：`build_backbone` 构造 DINOv3 权重，必要时加载已有微调参数，并通过 `DINOv3_Adapter` 导出 Mask2Former 需要的多尺度特征。
- **LoRA 微调**：在指定 transformer block 的 q/v 投影上注入 `LoRA_qkv`，仅更新少量低秩参数，使得冻结 backbone 仍可适配数据域。
- **Mask2Former 预测与损失**：`DinoMask2Former` 实例化 HF Mask2Former 架构，输出 query logits、副分支 mask，并计算匈牙利匹配主损失以及深度一致性损失。
- **优化与评估**：`train_one_epoch`/`evaluate` 负责 AMP、梯度累积、梯度裁剪、学习率调度、mIoU 评估与 checkpoint 管理；`export_predictions` 在训练结束后导出全尺寸预测。

## 2. 数据与输入预处理
训练脚本依据 `dataset` 配置构建 dataloader，核心约束包括：
- 图像与 mask 会被同步增广、归一化，并按 `ignore_index` 掩蔽无标签区域。
- `_build_targets` 将像素级 mask 整理为实例级 `mask_labels` 与 `class_labels`，供 Mask2FormerLoss 做匈牙利匹配。
- 校验阶段在 `_run_full_image_inference` 中对整幅图推理，再利用 `_resize_mask_to_size` 与 `cal_iou` 计算 mIoU。

## 3. DINOv3 主干与 LoRA 微调
### 3.1 Backbone 初始化与适配
`build_backbone` 会：
1. 解析 dinov3 相关配置并加载预训练权重；
2. 根据 `finetuned_weights`（如已有 DINOv3 微调结果）部分加载 state dict；
3. 将 backbone 置于 `eval` 模式并冻结梯度，以稳定特征统计；
4. 返回由 `DINOv3_Adapter` 封装的模块，使之按照 `mask2former.feature_order` 输出 {C2, C3, C4, C5} 风格的特征图。

### 3.2 LoRA 注入策略与数学描述
`MAS/model/Dino_LoRA.py` 对每个目标 self-attention block 的 `qkv` 线性层注入 `LoRA_qkv` 包装器，仅为 q、v 通道添加可训练的低秩更新，而 k 通道保持冻结。设原始线性层输出
$$
[q^\top, k^\top, v^\top]^\top = W_{qkv} x,\quad W_{qkv} \in \mathbb{R}^{3C \times d}.
$$
LoRA 通过 rank-$r$ 分解引入两个仿射映射 $(A_q, B_q)$ 和 $(A_v, B_v)$，得到
$$
\begin{aligned}
\Delta q &= \frac{r}{\alpha} B_q A_q x, \\
\Delta v &= \frac{r}{\alpha} B_v A_v x, \\
\tilde{q} &= q + \Delta q,\quad \tilde{v} = v + \Delta v,\quad \tilde{k} = k,
\end{aligned}
$$
其中 $\alpha$ 为 LoRA scaling，`dropout` 施加于 $x$ 以缓解过拟合。脚本通过 `train.lora_layers` 控制注入层（可选最后 $n$ 层、显式 layer 列表或全层），并在 `train_m2f.py` 中仅对 LoRA 参数开启梯度，保证主干其他权重保持冻结但仍支持 DDP。

## 4. Mask2Former 变换器解码与主损失
`DinoMask2Former` 复用 HF Mask2Former 的 transformer decoder。decoder 每层输出 `class_queries_logits`（形状 `[B, Q, num_classes+1]`，额外的 `no-object` 类保持匈牙利匹配所需）与 `masks_queries_logits`（`[B, Q, H, W]`），随后执行以下步骤：

1. **匈牙利匹配**：`Mask2FormerLoss` 构造匹配 cost（分类 cost + 点采样 BCE cost + Dice cost），利用匈牙利算法将预测 query 与 GT 实例一一对齐。
2. **主损失**：对匹配到的 query 计算
   $$
   \mathcal{L}_{cls} = -\frac{1}{N}\sum_{i=1}^{N} \log p_{i,y_i},
   $$
   $$
   \mathcal{L}_{mask} = -\frac{1}{NP}\sum_{i=1}^{N}\sum_{p=1}^{P} \big[g_{i,p}\log\sigma(m_{i,p}) + (1-g_{i,p})\log(1-\sigma(m_{i,p}))\big],
   $$
   $$
   \mathcal{L}_{dice} = 1 - \frac{2\sum_{p} g_{i,p}\sigma(m_{i,p})}{\sum_{p} g_{i,p}^2 + \sum_{p} \sigma(m_{i,p})^2},
   $$
   其中 $N$ 为匹配实例数，$P$ 为点采样位置数。多层 decoder 的辅助输出也重复该过程并按 `loss_weights` 加权。

## 5. 总体目标与训练策略
综合上述损失，单个 batch 的优化目标可以写为
$$
\mathcal{L}_{total} = \lambda_{cls}\mathcal{L}_{cls} + \lambda_{mask}\mathcal{L}_{mask} + \lambda_{dice}\mathcal{L}_{dice} + \sum_{\ell}\lambda_{\ell}^{aux}\mathcal{L}_{\ell}^{aux},
$$
其中 $\lambda$ 系数由 `DinoMask2Former.loss_weights` 与配置文件决定。`train_m2f.py` 采样的关键训练策略包括：
- **优化器**：对可训练参数（LoRA + Mask2Former 头）使用 `AdamW`，学习率、动量、权重衰减均来源于配置。
- **AMP 与梯度累积**：`torch.amp.autocast` + `GradScaler` 控制半精度训练，`grad_accum_steps` 支持跨多个 mini-batch 累积再反传，必要时执行 `clip_grad_norm_`。
- **调度器**：支持分数衰减、多步或余弦等策略，既可按 step 也可按 epoch 更新。
- **分布式同步**：依赖 `torch.distributed` 与 `DistributedDataParallel`，通过 `_distributed_mean` 汇总 loss 与 mIoU。

## 6. 日志、评估与导出
`init_training_logger` 负责 TensorBoard 记录与文本日志，关键指标含义如下：
- `loss`：所有加权 loss 的总和；
- `loss_mask / loss_dice / loss_cross_entropy`：主损失的未加权数值；
- `eval/miou`：验证集逐像素 mIoU。

训练过程中按 `eval_interval` 周期运行 `evaluate`，同时维护最佳 mIoU checkpoint。最后一个 epoch 后会：
1. 保存 `last.pt`、`best.pt`；
2. 调用 `export_predictions` 将验证集 mask 写入 `evaluation.pred_dir`；
3. 若提供 GT 目录，调用 `evaluate_prediction_directory` 计算最终指标，供论文复现实验引用。
