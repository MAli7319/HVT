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

    def generate(self, images, sos_idx, eos_idx, pad_idx, beam_size=5):
        """Autoregressive generation using Batched Beam Search."""
        B = images.size(0)
        device = images.device
        
        # 1. Encode Images
        full_enc_outputs = self.encoder(images)
        cls_token_out = full_enc_outputs[:, 0, :]    
        patch_outputs = full_enc_outputs[:, 1:, :]   
        patch_seq_len = patch_outputs.size(1)
        
        # Initialize LSTM states
        hidden = torch.tanh(self.init_h(cls_token_out)) 
        cell = torch.tanh(self.init_c(cls_token_out))   
        coverage = torch.zeros(B, patch_seq_len).to(device)
        
        # 2. Expand Tensors for Beam Search (B -> B * beam_size)
        patch_outputs = patch_outputs.unsqueeze(1).expand(B, beam_size, patch_seq_len, -1).contiguous().view(B * beam_size, patch_seq_len, -1)
        hidden = hidden.unsqueeze(1).expand(B, beam_size, -1).contiguous().view(B * beam_size, -1)
        cell = cell.unsqueeze(1).expand(B, beam_size, -1).contiguous().view(B * beam_size, -1)
        coverage = coverage.unsqueeze(1).expand(B, beam_size, -1).contiguous().view(B * beam_size, -1)
        
        # 3. Setup Beam Tracking Variables
        current_token = torch.full((B * beam_size,), sos_idx, dtype=torch.long).to(device)
        
        # Scores: (B, beam_size). First beam starts at 0, others at -infinity
        scores = torch.full((B, beam_size), -float('inf')).to(device)
        scores[:, 0] = 0.0 
        
        # Sequence history
        seqs = torch.full((B, beam_size, self.max_seq_len), pad_idx, dtype=torch.long).to(device)
        seqs[:, :, 0] = sos_idx
        
        # Track which beams have hit <EOS>
        finished = torch.zeros((B, beam_size), dtype=torch.bool).to(device)
        
        for t in range(1, self.max_seq_len):
            # Forward pass through decoder step
            logits, hidden, cell, coverage, _ = self.decoder_step(
                current_token, hidden, cell, patch_outputs, coverage
            ) 
            
            # Get log probabilities
            log_probs = torch.log_softmax(logits, dim=-1) 
            log_probs = log_probs.view(B, beam_size, self.vocab_size)
            
            # If a beam is finished, force it to only predict <EOS> with 0 penalty
            log_probs[finished] = -float('inf')
            log_probs[finished, eos_idx] = 0.0
            
            # Add to cumulative sequence scores
            next_scores = scores.unsqueeze(2) + log_probs 
            next_scores = next_scores.view(B, beam_size * self.vocab_size)
            
            # Find the top K scores across all vocabulary extensions
            topk_scores, topk_indices = torch.topk(next_scores, beam_size, dim=1)
            
            # Map the flat indices back to their specific beam and token
            beam_indices = topk_indices // self.vocab_size 
            token_indices = topk_indices % self.vocab_size 
            
            scores = topk_scores
            
            # 4. Gather the States for the Surviving Beams
            batch_indices = torch.arange(B).unsqueeze(1).expand(-1, beam_size).to(device)
            
            hidden = hidden.view(B, beam_size, -1)[batch_indices, beam_indices].view(B * beam_size, -1)
            cell = cell.view(B, beam_size, -1)[batch_indices, beam_indices].view(B * beam_size, -1)
            coverage = coverage.view(B, beam_size, -1)[batch_indices, beam_indices].view(B * beam_size, -1)
            
            seqs = seqs[batch_indices, beam_indices] 
            seqs[:, :, t] = token_indices
            
            finished = finished[batch_indices, beam_indices]
            finished = finished | (token_indices == eos_idx)
            
            current_token = token_indices.view(B * beam_size)
            
            # Early stopping if all beams in the entire batch have output <EOS>
            if finished.all():
                break
                
        # Return the #1 best sequence for each image (beam 0)
        best_seqs = seqs[:, 0, :]
        return best_seqs