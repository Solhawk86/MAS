# Loss 设计说明

## Mask2Former 主损失
- **原理**：沿用 HuggingFace Mask2Former 的匈牙利匹配策略，通过一对一匹配将预测 query 与 GT 实例对齐，再计算分类、点采样 BCE 和 Dice 三项损失。
- **class_queries_logits / masks_queries_logits 是什么**：
  1. Backbone 与 pixel decoder 生成多尺度特征，输入 transformer decoder。
  2. Transformer decoder 维护一组 depth-aware queries，经过多层 self + cross attention 后输出一系列 `intermediate_hidden_states`。每层 hidden state 通过线性层变换得到 `class_queries_logits`（形状 `[B, Q, num_classes+1]1是no-object类`）。
  3. 同时，decoder 将 hidden state 送入 mask embedding MLP，并与 mask features 做点积，得到 `masks_queries_logits`（形状 `[B, Q, H, W]`），即每个 query 对整幅图的 mask logit。
- **如何与 GT 做匈牙利匹配并得到预测 mask**：
  1. `Mask2FormerLoss` 会把 `class_queries_logits`、`masks_queries_logits` 与实例级 GT（类标签 + 二值 mask）一同输入。
  2. 模块内部先对 mask logits 做点采样，计算与 GT mask 的 pair-wise 损失，再综合分类 cost，构建代价矩阵。
  3. 使用匈牙利算法挑选匹配对，让每个 GT 实例只与一个 query 对应；未被匹配的 query 视为 “no object”。
  4. 对匹配后的 query，`masks_queries_logits` 经过 sigmoid 就是最终预测 mask，辅以后续的 BCE / Dice 计算。
- **计算流程**：
  1. `Mask2FormerLoss` 接收 `class_queries_logits`、`masks_queries_logits` 及 GT 标签，内部完成匈牙利匹配。
  2. 对匹配到的 query，分别计算：
     - 分类交叉熵：
       $$
       \mathcal{L}_{cls} = -\frac{1}{N} \sum_{i=1}^{N} \log p_{i,y_i}
       $$
       其中 $p_{i,y_i}$ 是第 $i$ 个匹配 query 对应类别 $y_i$ 的 softmax 概率。
     - 点采样 BCE：随机采样 $P$ 个点，将 mask logits 过 sigmoid 得到概率 $\sigma(m_{i,p})$，与 GT mask $g_{i,p}$ 做二元交叉熵：
       $$
       \mathcal{L}_{mask} = -\frac{1}{NP} \sum_{i=1}^{N} \sum_{p=1}^{P} \big[ g_{i,p} \log \sigma(m_{i,p}) + (1-g_{i,p}) \log (1-\sigma(m_{i,p})) \big]
       $$
     - Dice loss：
       $$
       \mathcal{L}_{dice} = 1 - \frac{2 \sum_{p} g_{i,p} \cdot \sigma(m_{i,p})}{\sum_{p} g_{i,p}^2 + \sum_{p} \sigma(m_{i,p})^2}
       $$
       对每个匹配实例求值后再平均。
  3. `DinoMask2Former` 通过 `self.loss_weights` 为三项主损失乘权，辅助层的 loss 也按同样方式加权再累加到总 loss。
- **涉及模块**：
  - `transformer_out.intermediate_hidden_states` / `.masks_queries_logits`（生成 logits）。
  - `Mask2FormerLoss`（`MAS/mask2former/model.py` 中实例化）。
  - 训练日志的 `loss_mask / loss_dice / loss_cross_entropy`（未加权数值）。

## 深度一致性损失
- **目标**：为 60 个 query 划分成 3 组（near/mid/far），通过图像清晰度约束 learnable depth embedding，使不同组预测不同深度范围。
- **计算流程**：
  1. **LaplacianClarity**：将输入图像反归一化到 0~1，转灰度后通过 3x3 拉普拉斯核求绝对值并归一化，得到 clarity map（`MAS/loss.py:132-154`）。
  2. **QueryDepthConsistencyLoss**（`MAS/loss.py:157-219`）：
     - 由 depth-aware latent 投影出 `query_depth` 标量。
     - 用预测 mask 对 clarity map 做加权平均，得到每个 query 的 `query_region_depth`：
       $$
       \hat{d_i} = \frac{\sum_{p} m_{i,p} \cdot c_p}{\sum_{p} m_{i,p} + 10^{-6}}
       $$
       其中 $m_{i,p}$ 是第 $i$ 个 query 在像素 $p$ 的 mask 概率，$c_p$ 是 clarity map 在像素 $p$ 的值。
     - 一致性项（L1）：
       $$
       \mathcal{L}_{depth} = \frac{1}{Q} \sum_{i=1}^{Q} \left| d_i - \hat{d_i} \right|
       $$
       其中 $d_i$ 是 learnable depth，$\hat{d_i}$ 是 mask 加权后的真实清晰度。
     - 组间多样性：对三组平均清晰度 $E_k$ 计算两两差的平方：
       $$
       \mathcal{L}_{div} = \sum_{k=1}^{3} \sum_{j=k+1}^{3} (E_k - E_j)^2
       $$
     - 总损失：$\mathcal{L}_{depth} + \lambda_{div} \mathcal{L}_{div}$。
  3. 在 `MAS/mask2former/model.py` 中，深度 loss 乘以 `depth_loss_weight` 与主损失相加，日志里记为 `depth_loss`（未加权）。
- **涉及模块**：
  - `MAS/depth.py`：`LearnableDepthQueryEmbedding` 负责产生 depth-aware query。
  - `MAS/mask2former/depth_transformer.py`：将 HF transformer 的 query embedder 替换为上面的实现，并输出 `depth_latent`/`depth_ids`。
  - `MAS/mask2former/model.py`：负责反归一化输入图像、调用 LaplacianClarity、汇总 depth loss。

## 训练日志
- `loss=`：所有 loss 加权后的总和。
- `loss_mask / loss_dice / loss_cross_entropy`：主损失三项未加权值。
- `depth_loss`：深度损失未加权值。

如需扩展，可在此基础上继续补充新的 loss 模块和使用方式。
