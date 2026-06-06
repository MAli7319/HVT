import math
import torch
import torch.nn as nn

""" Computing and adding continuous 2D sinusoidal positional encodings to preserve spatial relationships. """
class PositionalEncoding2D(nn.Module):
    """ Calculating and caching the 2D sinusoidal positional encoding tensor for height and width dimensions. """
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

    """ Adding the cached 2D positional encoding tensor to the input feature map. """
    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        return x + self.pe[:, :, :H, :W]

""" Flattening spatial feature grids into sequences of patch vectors for Transformer encoding. """
class FeatureFlattening(nn.Module):
    """ Initializing the FeatureFlattening module. """
    def __init__(self):
        super().__init__()

    """ Flattening height and width dimensions and transposing the tensor to batch-first sequence shape. """
    def forward(self, x):
        x = x.flatten(2) 
        x = x.permute(0, 2, 1) 
        return x

""" Implementing a Transformer encoder layer configured with Pre-LayerNorm topology. """
class CustomEncoderLayer(nn.Module):
    """ Setting up multi-head self-attention, layer normalizations, dropout, and feedforward layers. """
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

    """ Running Pre-LayerNorm self-attention and MLP blocks with residual connections on the source sequence. """
    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
            src_normed = self.norm1(src) 

            attn_output, _ = self.self_attn(
                src_normed, src_normed, src_normed,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask
            ) 

            src = src + self.dropout1(attn_output) 

            src_normed2 = self.norm2(src)

            ff_output = self.linear2(self.dropout(self.activation(self.linear1(src_normed2))))

            src = src + self.dropout2(ff_output)

            return src

""" Coordinating the CNN backbone, 2D positional encoding, CLS token prepending, and Transformer layers. """
class HVTEncoder(nn.Module):
    """ Setting up the ResNet backbone, 2D positional encoding, flattening, CLS token, and Transformer encoder. """
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

    """ Processing raw images into encoded sequence vectors and prepending the learnable CLS token. """
    def forward(self, images):
        features = self.backbone(images) 
        features = self.pos_encoder(features)
        seq = self.flatten(features)

        B = seq.shape[0] 
        cls_tokens = self.cls_token.expand(B, -1, -1) 
        seq = torch.cat((cls_tokens, seq), dim=1)

        encoder_output = self.transformer_encoder(seq)
        return encoder_output

