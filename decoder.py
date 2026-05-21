import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 3. LSTM DECODER COMPONENTS
# ==========================================
class CoverageAttention(nn.Module):
    def __init__(self, encoder_dim=512, decoder_dim=256, attention_dim=256):
        super().__init__()
        self.W_enc = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.W_dec = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.W_cov = nn.Conv1d(in_channels=1, out_channels=attention_dim, 
                               kernel_size=5, padding=2, bias=False)
        self.V = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, coverage):
        enc_features = self.W_enc(encoder_outputs) 
        dec_features = self.W_dec(hidden).unsqueeze(1) 
        
        cov_reshaped = coverage.unsqueeze(1) 
        cov_features = self.W_cov(cov_reshaped).permute(0, 2, 1) 
        
        combined_features = torch.tanh(enc_features + dec_features + cov_features)
        attention_scores = self.V(combined_features).squeeze(2)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        next_coverage = coverage + attention_weights
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context_vector, attention_weights, next_coverage


class HVTDecoderStep(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, encoder_dim=512, decoder_dim=256, attention_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = CoverageAttention(encoder_dim, decoder_dim, attention_dim)
        self.lstm = nn.LSTMCell(input_size=(embed_dim + encoder_dim), hidden_size=decoder_dim)
        self.classifier = nn.Linear(decoder_dim + encoder_dim, vocab_size)

    def forward(self, prev_token, hidden, cell, encoder_outputs, coverage):
        embedded = self.embedding(prev_token)
        context, attn_weights, next_coverage = self.attention(hidden, encoder_outputs, coverage)
        
        lstm_input = torch.cat([embedded, context], dim=1)
        next_hidden, next_cell = self.lstm(lstm_input, (hidden, cell))
        
        classifier_input = torch.cat([next_hidden, context], dim=1)
        prediction_logits = self.classifier(classifier_input)
        
        return prediction_logits, next_hidden, next_cell, next_coverage, attn_weights
