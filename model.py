import torch
import torch.nn as nn
import random

from backbone import MathResNetBackbone
from encoder import HVTEncoder
from decoder import HVTDecoderStep

""" Wrapping the entire Seq2Seq pipeline including the HVT encoder and HVT decoder step. """
class HVTSeq2Seq(nn.Module):
    """ Initializing the ResNet backbone, HVT encoder, LSTM decoder step, and state initialization projections. """
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

    """ Running the training forward pass using teacher forcing to predict next tokens in the sequence. """
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

        """ Autoregressively decoding tokens step-by-step up to the maximum target sequence length. """
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

    """ Generating predictions autoregressively using a batched beam search decoder. """
    def generate(self, images, sos_idx, eos_idx, pad_idx, beam_size=5):
        B = images.size(0)
        device = images.device

        full_enc_outputs = self.encoder(images)
        cls_token_out = full_enc_outputs[:, 0, :]    
        patch_outputs = full_enc_outputs[:, 1:, :]   
        patch_seq_len = patch_outputs.size(1)

        hidden = torch.tanh(self.init_h(cls_token_out)) 
        cell = torch.tanh(self.init_c(cls_token_out))   
        coverage = torch.zeros(B, patch_seq_len).to(device)

        patch_outputs = patch_outputs.unsqueeze(1).expand(B, beam_size, patch_seq_len, -1).contiguous().view(B * beam_size, patch_seq_len, -1)
        hidden = hidden.unsqueeze(1).expand(B, beam_size, -1).contiguous().view(B * beam_size, -1)
        cell = cell.unsqueeze(1).expand(B, beam_size, -1).contiguous().view(B * beam_size, -1)
        coverage = coverage.unsqueeze(1).expand(B, beam_size, -1).contiguous().view(B * beam_size, -1)

        current_token = torch.full((B * beam_size,), sos_idx, dtype=torch.long).to(device)

        scores = torch.full((B, beam_size), -float('inf')).to(device)
        scores[:, 0] = 0.0 

        seqs = torch.full((B, beam_size, self.max_seq_len), pad_idx, dtype=torch.long).to(device)
        seqs[:, :, 0] = sos_idx

        finished = torch.zeros((B, beam_size), dtype=torch.bool).to(device)

        """ Decoding step-by-step to update top beam candidates based on cumulative log probabilities. """
        for t in range(1, self.max_seq_len):
            logits, hidden, cell, coverage, _ = self.decoder_step(
                current_token, hidden, cell, patch_outputs, coverage
            ) 

            log_probs = torch.log_softmax(logits, dim=-1) 
            log_probs = log_probs.view(B, beam_size, self.vocab_size)

            log_probs.masked_fill_(finished.unsqueeze(-1), -float('inf'))
            log_probs[:, :, eos_idx].masked_fill_(finished, 0.0)

            next_scores = scores.unsqueeze(2) + log_probs 
            next_scores = next_scores.view(B, beam_size * self.vocab_size)

            topk_scores, topk_indices = torch.topk(next_scores, beam_size, dim=1)

            beam_indices = topk_indices // self.vocab_size 
            token_indices = topk_indices % self.vocab_size 

            scores = topk_scores

            batch_indices = torch.arange(B).unsqueeze(1).expand(-1, beam_size).to(device)

            hidden = hidden.view(B, beam_size, -1)[batch_indices, beam_indices].view(B * beam_size, -1)
            cell = cell.view(B, beam_size, -1)[batch_indices, beam_indices].view(B * beam_size, -1)
            coverage = coverage.view(B, beam_size, -1)[batch_indices, beam_indices].view(B * beam_size, -1)

            seqs = seqs[batch_indices, beam_indices] 
            seqs[:, :, t] = token_indices

            finished = finished[batch_indices, beam_indices]
            finished = finished | (token_indices == eos_idx)

            current_token = token_indices.view(B * beam_size)

            if finished.all():
                break

        best_seqs = seqs[:, 0, :]
        return best_seqs

