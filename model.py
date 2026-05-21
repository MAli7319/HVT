import torch
import torch.nn as nn
import random

from backbone import MathResNetBackbone
from encoder import HVTEncoder
from decoder import HVTDecoderStep

# ==========================================
# FINAL SEQ2SEQ WRAPPER
# ==========================================
class HVTSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=512, decoder_dim=256, max_seq_len=150):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        self.backbone = MathResNetBackbone()
        self.encoder = HVTEncoder(self.backbone, d_model=d_model)
        
        self.init_h = nn.Linear(d_model, decoder_dim)
        self.init_c = nn.Linear(d_model, decoder_dim)
        
        self.decoder_step = HVTDecoderStep(vocab_size=vocab_size, 
                                           encoder_dim=d_model, 
                                           decoder_dim=decoder_dim)

    def forward(self, images, target_seqs=None, teacher_forcing_ratio=0.5):
        B = images.size(0)
        device = images.device
        
        full_enc_outputs = self.encoder(images)
        cls_token_out = full_enc_outputs[:, 0, :]    
        patch_outputs = full_enc_outputs[:, 1:, :]   
        patch_seq_len = patch_outputs.size(1)
        
        hidden = torch.tanh(self.init_h(cls_token_out)) 
        cell = torch.tanh(self.init_c(cls_token_out))   
        
        coverage = torch.zeros(B, patch_seq_len).to(device)
        outputs = torch.zeros(B, self.max_seq_len, self.vocab_size).to(device)
        
        current_token = torch.zeros(B, dtype=torch.long).to(device) 
        if target_seqs is not None:
             current_token = target_seqs[:, 0]
        
        for t in range(1, self.max_seq_len):
            logits, hidden, cell, coverage, _ = self.decoder_step(
                current_token, hidden, cell, patch_outputs, coverage
            )
            
            outputs[:, t, :] = logits
            top1_prediction = logits.argmax(dim=1)
            
            if target_seqs is not None and random.random() < teacher_forcing_ratio:
                current_token = target_seqs[:, t] 
            else:
                current_token = top1_prediction 
                
        return outputs