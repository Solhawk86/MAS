import torch
import torch.nn as nn


class LearnableDepthQueryEmbedding(nn.Module):
    """Mask2Former query embedding + 3 组深度向量."""

    def __init__(self, num_queries: int, hidden_dim: int, num_depth: int = 3):
        super().__init__()
        if num_queries % num_depth != 0:
            raise ValueError(
                f"num_queries({num_queries}) 必须能被 depth 组数({num_depth})整除"
            )
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_depth = num_depth

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.group_depth_embed = nn.Embedding(num_depth, hidden_dim)
        self.depth_scale = nn.Parameter(torch.tensor(0.1))

        self._reset_parameters()
        depth_ids = self._build_depth_ids(num_queries, num_depth)
        self.register_buffer("depth_ids", depth_ids, persistent=False)

    def _reset_parameters(self):
        nn.init.zeros_(self.group_depth_embed.weight)
        nn.init.uniform_(self.query_embed.weight)

    @staticmethod
    def _build_depth_ids(num_queries: int, num_depth: int) -> torch.Tensor:
        # 将 query 依次映射到 0~num_depth-1 组
        # 例如：
        # group = 6 // 3 = 2
        # ids = [0,0,1,1,2,2]
        group = num_queries // num_depth
        ids = torch.arange(num_queries, dtype=torch.long) // group
        return ids

    def _depth_latent(self) -> torch.Tensor:
        return self.group_depth_embed(self.depth_ids)

    def _query_latent(self) -> torch.Tensor:
        return self.query_embed.weight + self.depth_scale * self._depth_latent()

    @property
    def weight(self) -> torch.Tensor:
        """向下兼容 nn.Embedding.weight 接口，供 HF decoder 直接读取。"""
        return self._query_latent()

    def get_query_embedding(self, batch_size: int) -> torch.Tensor:
        return self._query_latent().unsqueeze(1).expand(-1, batch_size, -1)

    def get_depth_latent(self) -> torch.Tensor:
        return self._depth_latent()

    def forward(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.get_query_embedding(batch_size), self.depth_ids
