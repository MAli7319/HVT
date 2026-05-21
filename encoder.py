import math
import torch
import torch.nn as nn

# ==========================================
# 2. TRANSFORMER ENCODER COMPONENTS
# ==========================================
class PositionalEncoding2D(nn.Module):
    """The missing continuous sinusoidal 2D PE module"""
    def __init__(self, d_model=512, max_h=128, max_w=512):
        super().__init__()
        d_half = d_model // 2 
        
        y_pos = torch.arange(max_h, dtype=torch.float).unsqueeze(1)
        x_pos = torch.arange(max_w, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_half, 2).float() * (-math.log(10000.0) / d_half))
        
        pe_y = torch.zeros(max_h, d_half)
        pe_y[:, 0::2] = torch.sin(y_pos * div_term) 
        pe_y[:, 1::2] = torch.cos(y_pos * div_term) 
        
        pe_x = torch.zeros(max_w, d_half)
        pe_x[:, 0::2] = torch.sin(x_pos * div_term)
        pe_x[:, 1::2] = torch.cos(x_pos * div_term)
        
        pe_y = pe_y.unsqueeze(1).expand(-1, max_w, -1)
        pe_x = pe_x.unsqueeze(0).expand(max_h, -1, -1)
        
        pe = torch.cat([pe_y, pe_x], dim=-1)
        pe = pe.permute(2, 0, 1).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        return x + self.pe[:, :, :H, :W]


class FeatureFlattening(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.flatten(2) 
        x = x.permute(0, 2, 1) 
        return x


class CustomEncoderLayer(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
            # ----------------------------------------------------
            # Block 1: Pre-Norm -> Self-Attention -> Residual Add
            # ----------------------------------------------------
            # 1. Apply LayerNorm FIRST
            src_normed = self.norm1(src) 
            
            # 2. Attention uses the normed input (and passes the masks along)
            attn_output, _ = self.self_attn(
                src_normed, src_normed, src_normed,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask
            ) 
            
            # 3. Residual adds to the ORIGINAL un-normed src
            src = src + self.dropout1(attn_output) 

            # ----------------------------------------------------
            # Block 2: Pre-Norm -> FeedForward -> Residual Add
            # ----------------------------------------------------
            # 1. Apply LayerNorm FIRST
            src_normed2 = self.norm2(src)
            
            # 2. MLP uses the normed input
            ff_output = self.linear2(self.dropout(self.activation(self.linear1(src_normed2))))
            
            # 3. Residual adds to the ORIGINAL un-normed src
            src = src + self.dropout2(ff_output)

            return src


class HVTEncoder(nn.Module):
    def __init__(self, backbone, d_model=512, nhead=8, num_layers=4):
        super().__init__()
        self.backbone = backbone 
        self.pos_encoder = PositionalEncoding2D(d_model)
        self.flatten = FeatureFlattening()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.transformer_encoder = nn.TransformerEncoder(
            CustomEncoderLayer(d_model, nhead), 
            num_layers=num_layers,
            enable_nested_tensor=False
        )

    def forward(self, images):
        features = self.backbone(images) 
        features = self.pos_encoder(features)
        seq = self.flatten(features)
        
        B = seq.shape[0] 
        cls_tokens = self.cls_token.expand(B, -1, -1) 
        seq = torch.cat((cls_tokens, seq), dim=1)
        
        encoder_output = self.transformer_encoder(seq)
        return encoder_output
