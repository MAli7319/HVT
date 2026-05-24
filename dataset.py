import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class MathTokenizer:
    def __init__(self, pad_token="<PAD>", sos_token="<SOS>", eos_token="<EOS>", unk_token="<UNK>"):
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        
        self.word2idx = {pad_token: 0, sos_token: 1, eos_token: 2, unk_token: 3}
        self.idx2word = {0: pad_token, 1: sos_token, 2: eos_token, 3: unk_token}
        self.vocab_size = 4

    def build_vocab(self, formulas):
        """Builds vocabulary from a list of space-separated formula strings."""
        for formula in formulas:
            tokens = formula.strip().split()
            for token in tokens:
                if token not in self.word2idx:
                    self.word2idx[token] = self.vocab_size
                    self.idx2word[self.vocab_size] = token
                    self.vocab_size += 1

    def encode(self, formula, max_length):
        """Converts a string formula into a padded sequence of integer IDs."""
        tokens = formula.strip().split()
        
        # Start with SOS
        sequence = [self.word2idx[self.sos_token]]
        
        # Add formula tokens (replace unknowns with UNK)
        for token in tokens:
            sequence.append(self.word2idx.get(token, self.word2idx[self.unk_token]))
            
        # Add EOS
        sequence.append(self.word2idx[self.eos_token])
        
        # Truncate if too long (leave room for EOS)
        if len(sequence) > max_length:
            sequence = sequence[:max_length-1] + [self.word2idx[self.eos_token]]
            
        # Pad if too short
        while len(sequence) < max_length:
            sequence.append(self.word2idx[self.pad_token])
            
        return torch.tensor(sequence, dtype=torch.long)


class Im2LatexDataset(Dataset):
    def __init__(self, data_dir, split_csv, tokenizer, max_seq_len=150, transform=None):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "formula_images_processed")
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.transform = transform
        
        split_path = os.path.join(data_dir, split_csv)
        
        # The exact Pandas config to handle quoted LaTeX strings
        self.metadata = pd.read_csv(
            split_path, 
            header=0,           # FIX: Treat row 0 as column names and skip it!
            sep=',',            
            quotechar='"',      
            names=['formula', 'image'],
            on_bad_lines='skip' 
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        image_name = str(row['image']).strip()
        if not image_name.endswith('.png'):
            image_name += '.png'
            
        image_path = os.path.join(self.images_dir, image_name)
        
        try:
            image = Image.open(image_path).convert('L')
        except FileNotFoundError:
            image = Image.new('L', (512, 128), color=255)
            
        if self.transform:
            image = self.transform(image)
            
        # We extract the formula string directly from the row!
        formula_str = str(row['formula'])
        
        target_seq = self.tokenizer.encode(formula_str, self.max_seq_len)
        
        return image, target_seq
