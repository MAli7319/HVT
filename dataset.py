import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

""" Implementing a tokenizer that builds a vocabulary mapping LaTeX tokens to integer IDs and encodes formula strings. """
class MathTokenizer:
    """ Initializing the tokenizer vocabulary mappings with special pad, sos, eos, and unk tokens. """
    def __init__(self, pad_token="<PAD>", sos_token="<SOS>", eos_token="<EOS>", unk_token="<UNK>"):
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        self.word2idx = {pad_token: 0, sos_token: 1, eos_token: 2, unk_token: 3}
        self.idx2word = {0: pad_token, 1: sos_token, 2: eos_token, 3: unk_token}
        self.vocab_size = 4

    """ Building the vocabulary by parsing and registering unique tokens from all training formula strings. """
    def build_vocab(self, formulas):
        """ Iterating through each formula in the training dataset to extract tokens and build the vocabulary. """
        for formula in formulas:
            tokens = formula.strip().split()
            """ Registering new tokens to the vocabulary mapping if they are not already present. """
            for token in tokens:
                if token not in self.word2idx:
                    self.word2idx[token] = self.vocab_size
                    self.idx2word[self.vocab_size] = token
                    self.vocab_size += 1

    """ Converting a raw LaTeX formula string into a padded integer tensor of vocabulary IDs. """
    def encode(self, formula, max_length):
        tokens = formula.strip().split()

        sequence = [self.word2idx[self.sos_token]]

        """ Appending vocabulary IDs of the formula tokens and falling back to UNK for out-of-vocabulary tokens. """
        for token in tokens:
            sequence.append(self.word2idx.get(token, self.word2idx[self.unk_token]))

        sequence.append(self.word2idx[self.eos_token])

        if len(sequence) > max_length:
            sequence = sequence[:max_length-1] + [self.word2idx[self.eos_token]]

        while len(sequence) < max_length:
            sequence.append(self.word2idx[self.pad_token])

        return torch.tensor(sequence, dtype=torch.long)

""" Implementing a dataset loader that reads LaTeX formula labels and image paths from a split configuration CSV. """
class Im2LatexDataset(Dataset):
    """ Loading and filtering the split CSV, setting up image directories, transforms, and the tokenizer. """
    def __init__(self, data_dir, split_csv, tokenizer, max_seq_len=150, transform=None):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "formula_images_processed")
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.transform = transform

        split_path = os.path.join(data_dir, split_csv)

        self.metadata = pd.read_csv(
            split_path, 
            header=0,
            sep=',',            
            quotechar='"',      
            names=['formula', 'image'],
            on_bad_lines='skip' 
        )

    """ Returning the total number of samples available in the dataset. """
    def __len__(self):
        return len(self.metadata)

    """ Loading a formula image, applying transforms, and encoding its corresponding LaTeX formula string. """
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

        formula_str = str(row['formula'])

        target_seq = self.tokenizer.encode(formula_str, self.max_seq_len)

        return image, target_seq

""" Batching collated images by padding variable-sized shapes to the maximum dimensions in the batch. """
def pad_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    max_h = max([img.shape[1] for img in images])
    max_w = max([img.shape[2] for img in images])

    padded_images = []
    """ Looping over all images in the batch to apply padding up to the maximum batch height and width. """
    for img in images:
        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]

        padded = F.pad(img, (0, pad_w, 0, pad_h), value=1.0)
        padded_images.append(padded)

    return torch.stack(padded_images), torch.stack(targets)

