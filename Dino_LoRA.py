import math
import torch    
from torch import nn    

class LoRA_qkv(nn.module):
    def __init__(self, base_modle, rank: int, alpha: float, dropout: int = 0):
        super().__init__()
        if rank <=0 :
            raise ValueError("rank must bigger than 0")
        
        self.base_modle = base_modle
        self.rank = rank
        self.alpha = alpha
        self.scaling = rank / alpha
        self.dropout = nn.Dropout(dropout)
        
        for p in base_modle.parameters():
            p.requires_grad = False
          
        device = self.base_modle.weight.device
        dtype = self.base_modle.weight.dtype  
        
        self.lora_A = nn.Linear(self.base_modle.in_features, rank , bias=False)
        self.lora_B = nn.Linear(rank, self.base_modle.in_features , bias=False)
        
        #初始化LoRA权重
        nn.init.kaiming_uniform_(self.lora_A.weight, a = math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.lora_A.to(device= device, dtype=dtype)
        self.lora_B.to(device= device, dtype=dtype)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # y=W′x=(W+(r/α)​BA)x=Wx+(r/α)​B(Ax)
        base_out = self.base_modle(x)
        lora_out = (self.lora_B(self.lora_A(self.dropout(x)))) * self.scaling
        return base_out + lora_out
        
        
        
        
        
